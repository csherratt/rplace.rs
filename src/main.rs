extern crate clap;
extern crate csv;
extern crate glutin;
#[macro_use]
extern crate gfx;
extern crate gfx_core;
extern crate gfx_device_gl;
extern crate gfx_window_glutin;

use std::collections::HashMap;

use clap::{Arg, App};
use gfx::traits::{Device, Factory, FactoryExt};
use gfx_core::format::{DepthStencil, Rgba8};

// the maximum length of a linked list of changes
const MAX_CHANGE_LL_LENGTH: usize = 256;
const MAX_CHANGES: usize = 2048;

struct Tile {
    /// tile offset x
    x: u16,
    /// tile offset y
    y: u16,
    /// timestamp (of 0)
    timestamp: u32,
    /// last used square
    tail: usize,

    /// 4 bits for the colour
    /// 12 bits for the timestamp (delta)
    /// [colour, 31..28][next, 27..17][timestamp, 16..0]
    changes: [u32; MAX_CHANGES],
}

pub struct Change(u32);

impl Change {
    fn new() -> Change {
        Change(0)
    }

    fn set_next(self, next: usize) -> Change {
        let next = next as u32;
        let new = self.0 & 0xF001_FFFF;
        Change(new | ((next & 0x7FF) << 17))
    }

    fn next(self) -> usize {
        ((self.0 >> 17) & 0x7FF) as usize
    }

    fn set_timestamp(self, ts: u32) -> Change {
        let new = self.0 & 0xFFF7_0000;
        Change(new | (ts & 0x1FFFF))
    }

    fn timestamp(self) -> u32 {
        (self.0 & 0xFFFF) as u32
    }

    fn set_colour(self, colour: u8) -> Change {
        let colour = colour as u32;
        let new = self.0 & 0x0FFF_FFFF;
        Change(new | ((colour & 0xF) << 28))
    }
}

impl Tile {
    pub fn new(x: u16, y: u16, ts: u32) -> Tile {
        Tile {
            x: x,
            y: y,
            timestamp: ts,
            tail: 256,
            changes: [0; MAX_CHANGES],
        }
    }

    fn newest_timestamp(&self) -> u32 {
        let mut ts = 0;
        for &change in &self.changes[..] {
            let next = Change(change).timestamp();
            ts = std::cmp::max(ts, next);
        }
        ts + self.timestamp
    }

    pub fn derive_from(src: &Tile) -> Tile {
        let mut new = Tile::new(src.x, src.y, src.newest_timestamp());
        for i in 0..256 {
            let (last, _) = src.seek_end(i);
            new.changes[i] = Change(src.changes[last])
                .set_timestamp(0)
                .set_next(0).0;
        }
        new
    }

    /// seek the end of a linked list chain, returning the index
    fn seek_end(&self, mut link: usize) -> (usize, usize) {
        let mut len = 1;
        loop {
            let change = Change(self.changes[link]);
            let next = change.next();
            if next == 0 {
                break;
            }
            link = next;
            len += 1;
        }
        (link, len)
    }

    pub fn append(&mut self, x: u16, y: u16, ts: u32, colour: u8) -> bool {
        // no space, skip, or the timestamp is to big to fit
        if self.tail >= MAX_CHANGES || ts - self.timestamp >= 0x1_0000 {
            return false;
        }

        let (x, y) = (x & 0xF, y & 0xF);
        let (index, len) = self.seek_end((y * 16 + x) as usize);

        // if the chain is to long we don't want to stall the GPU,
        // better to make a new tile rather then keep appending to
        // the same one.
        if len > MAX_CHANGE_LL_LENGTH {
            return false;
        }

        let change = Change(self.changes[index]);
        let tail = self.tail;
        self.tail += 1;

        // set the index of the next pointer
        self.changes[index] = change.set_next(tail).0;
        // set the new cell with the correct values
        self.changes[tail] = Change::new()
            .set_timestamp(ts - self.timestamp)
            .set_colour(colour)
            .0;
        true
    }
}

fn read_csv(path: &str) -> HashMap<(u16, u16), Vec<Tile>> {
    let mut latest = HashMap::new();
    let mut tiles = HashMap::new();

    let mut last_ts = 0;

    let mut rdr = csv::Reader::from_file(path).unwrap();
    for record in rdr.decode() {
        let (ts, _, y, x, colour): (u64, String, u16, u16, u8) = record.unwrap();
        let ts = (ts / 1000) as u32;
        if ts < last_ts {
            panic!("backward timestamp, ts={}, last_ts={}", ts, last_ts);
        }
        last_ts = ts;
        let tile_key = (x & 0xFFF0, y & 0xFFF0);
        while {
            let mut tile = latest.entry(tile_key)
                .or_insert_with(|| Tile::new(tile_key.0, tile_key.1, ts));
            !tile.append(x, y, ts, colour)
        } {
            let mut tile = tiles.entry(tile_key).or_insert_with(Vec::new);
            let last = latest.remove(&tile_key).unwrap();
            latest.insert(tile_key, Tile::derive_from(&last));
            tile.push(last);
        }
    }

    for (key, t) in latest {
        let mut tile = tiles.entry(key).or_insert_with(Vec::new);
        tile.push(t);
    }

    tiles
}

pub static VERTEX_SHADER_SRC: &'static [u8] = b"
    #version 150 core

    //uniform mat4 u_Proj;
    //uniform mat4 u_View;

    in vec2 a_Pos;
    in uvec2 a_Coord;

    out VertexData {
        vec2 Coord;
    } v_Out;

    void main() {
        v_Out.Coord = vec2(a_Coord);
        gl_Position = /*u_Proj * u_View  */ vec4(a_Pos, 0., 1.);
    }
";

pub static FRAGMENT_SHADER_SRC: &'static [u8] = b"
    #version 150 core

    uniform usampler1D t_Changes;

    struct Pallet {
        vec4 colour;
    };

    uniform b_Pallet {
        Pallet pallet[16];
    };

    uniform int u_Timestamp;

    in VertexData {
        vec2 Coord;
    } v_In;

    out vec4 o_Colour;

    void main() {
        uint x = uint(v_In.Coord.x);
        uint y = uint(v_In.Coord.y);
        uint pallet_idx = uint(0);
        uint idx = y * uint(16) + x;

        // avoid looping forever if something fucks up
        for (int i = 0; i <= 16; i++) {
            uint c = texelFetch(t_Changes, int(idx), 0).x;
            uint next = (c >> uint(17)) & uint(0x7FF);
            int ts = int(c) & 0x1FFFF;

            if (ts > u_Timestamp) {
                break;
            }
            pallet_idx = (c >> uint(28)) & uint(0xF);

            // no next, this is it
            if (next == uint(0)) {
                break;
            }
            idx = next;
        }

        o_Colour = pallet[pallet_idx].colour;
    }
";


gfx_defines!{
    vertex Vertex {
        pos: [f32; 2] = "a_Pos",
        coord: [u8; 2] = "a_Coord",
    }

    constant GfxChange {
        change: u32 = "change",
    }

    constant Pallet {
        colour: [f32; 4] = "colour",
    }

    pipeline pipe {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        tile: gfx::TextureSampler<u32> = "t_Changes",
        pallet: gfx::ConstantBuffer<Pallet> = "b_Pallet",

        //proj: gfx::Global<[[f32; 4]; 4]> = "u_Proj",
        //view: gfx::Global<[[f32; 4]; 4]> = "u_View",
        time: gfx::Global<i32> = "u_Timestamp",

        out_colour: gfx::RenderTarget<Rgba8> = "o_Colour",
        out_depth: gfx::DepthTarget<DepthStencil> =
            gfx::preset::depth::LESS_EQUAL_WRITE,
    }
}

// find the offset of the tile into the mesh buffer
fn mesh_offset(x: u16, y: u16) -> u32 {
    let (x, y) = (x as u32, y as u32);
    (x * 64 + y) * 6
}

// create the mesh
fn build_mesh() -> Vec<Vertex> {
    let scale = |pt: u32| -> f32 { ((pt as f32) - 32.) / 32. };

    let mut mesh = Vec::new();
    for y in 0..(1024 / 16) {
        for x in 0..(1024 / 16) {
            let pt0 = Vertex {
                pos: [scale(x), -scale(y)],
                coord: [0, 0],
            };
            let pt1 = Vertex {
                pos: [scale(x), -scale(y + 1)],
                coord: [16, 0],
            };
            let pt2 = Vertex {
                pos: [scale(x + 1), -scale(y + 1)],
                coord: [16, 16],
            };
            let pt3 = Vertex {
                pos: [scale(x + 1), -scale(y)],
                coord: [0, 16],
            };

            // triangle 0
            mesh.push(pt0);
            mesh.push(pt1);
            mesh.push(pt2);

            // triangle 1
            mesh.push(pt2);
            mesh.push(pt3);
            mesh.push(pt0);
        }
    }
    mesh
}

fn decode_pallet(colour: u32) -> Pallet {
    Pallet {
        colour: [(((colour >> 16) & 0xFF) as f32 / 255.),
                 (((colour >> 8) & 0xFF) as f32 / 255.),
                 (((colour >> 0) & 0xFF) as f32 / 255.),
                 1.],
    }
}

fn main() {
    let matches = App::new("/r/place viewer")
        .about("A viewer for reddit.com/r/place")
        .arg(Arg::with_name("INPUT")
            .help("Sets the csv file to use")
            .required(true)
            .index(1))
        .get_matches();

    println!("Loading tiles from {}, this can be slow...",
             matches.value_of("INPUT").unwrap());
    let tiles = read_csv(matches.value_of("INPUT").unwrap());

    println!("tiles loaded");

    let gl_version = glutin::GlRequest::Specific(glutin::Api::OpenGl, (3, 2));
    let builder = glutin::WindowBuilder::new()
        .with_gl(gl_version)
        .with_title("/r/place viewer".to_string())
        .with_dimensions(800, 800);

    let (window, mut device, mut factory, rtv, stv) =
        gfx_window_glutin::init::<Rgba8, DepthStencil>(builder);

    let mesh = build_mesh();
    let mesh = factory.create_vertex_buffer(&mesh);
    let combuf = factory.create_command_buffer();
    let mut encoder: gfx::Encoder<gfx_device_gl::Resources, _> = combuf.into();

    let sinfo = gfx::texture::SamplerInfo::new(
        gfx::texture::FilterMethod::Scale,
        gfx::texture::WrapMode::Clamp);
    let sampler = factory.create_sampler(sinfo);

    let pso = factory.create_pipeline_simple(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, pipe::new())
        .expect("Could not create PSO for `DrawShaded`!");
    let pallet_data = vec![decode_pallet(0xFFFFFF),
                           decode_pallet(0xEFEFEF),
                           decode_pallet(0x888888),
                           decode_pallet(0x222222),
                           decode_pallet(0xFFA7D1),
                           decode_pallet(0xE50000),
                           decode_pallet(0xE59500),
                           decode_pallet(0xA06A42),
                           decode_pallet(0xE5D900),
                           decode_pallet(0x94E044),
                           decode_pallet(0x02BE01),
                           decode_pallet(0x00D3DD),
                           decode_pallet(0x0083C7),
                           decode_pallet(0x0000EA),
                           decode_pallet(0xCF6EE4),
                           decode_pallet(0x820080)];
    let pallet = factory.create_buffer_immutable(&pallet_data, gfx::buffer::Role::Constant, gfx::Bind::empty()).unwrap();

    println!("Loading to GPU memory");
    let mut gpu_tiles_buffers = HashMap::new();
    // write all the tiles to the gpu
    for ((x, y), tiles) in tiles {
        let mut buffers = Vec::new();
        for tile in tiles {
            let kind = gfx::texture::Kind::D1(2048);
            let data = [&tile.changes[..]; 1];
            let (_, buffer) = factory.create_texture_immutable::<(gfx::format::R32, gfx::format::Uint)>(kind, &data[..]).unwrap();
            buffers.push((tile.timestamp, buffer));
        }
        gpu_tiles_buffers.insert((x, y), buffers);
    }

    println!("ready");
    let mut ts = 1490931573;
    loop {
        encoder.clear(&rtv, [1., 1., 1., 0.]);
        encoder.clear_depth(&stv, 1.);

        // draw all the tiles
        for (&(x, y), tile) in &gpu_tiles_buffers {
            let mut idx = None;
            for (i, &(timestamp, _)) in tile.iter().enumerate() {
                if timestamp >= ts {
                    break;
                }
                idx = Some(i);
            }

            let idx = if let Some(idx) = idx { idx } else { continue; };

            let off = mesh_offset(x / 16, y / 16);
            let slice = gfx::Slice {
                start: off,
                end: off + 6,
                base_vertex: 0,
                instances: None,
                buffer: gfx::IndexBuffer::Auto,
            };
            let mut delta = (ts - tile[idx].0) as i32;
            if delta < 0 {
                delta = 0;
            }

            let data = pipe::Data {
                vbuf: mesh.clone(),
                tile: (tile[idx].1.clone(), sampler.clone()),
                pallet: pallet.clone(),
                time: delta,
                out_colour: rtv.clone(),
                out_depth: stv.clone(),
            };
            encoder.draw(&slice, &pso, &data);
        }

        // send the command queue to the device
        encoder.flush(&mut device);

        // swap buffers
        window.swap_buffers().unwrap();
        device.cleanup();

        // only draw on events
        for _ in window.poll_events() {}

        ts += 60;
    }
}
