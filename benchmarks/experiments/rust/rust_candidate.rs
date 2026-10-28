//! Experimental sequential Rust kernel for the maximum-dissimilarity algorithm.
//!
//! The input is a small binary interchange format written by `evaluate.py`.
//! This script intentionally uses only Rust's standard library.

use std::env;
use std::fs;
use std::process;
use std::time::Instant;

const MAGIC: &[u8; 8] = b"MDAPYRS1";

struct Input<T> {
    rows: usize,
    columns: usize,
    selections: usize,
    seed_index: usize,
    directional: Vec<bool>,
    deviations: Vec<f64>,
    data: Vec<T>,
}

trait DataValue: Copy {
    fn difference(left: Self, right: Self) -> f64;
}

impl DataValue for f64 {
    #[inline]
    fn difference(left: Self, right: Self) -> f64 {
        left - right
    }
}

impl DataValue for f32 {
    #[inline]
    fn difference(left: Self, right: Self) -> f64 {
        (left - right) as f64
    }
}

fn read_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, String> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| "input offset overflow".to_string())?;
    let chunk = bytes
        .get(*offset..end)
        .ok_or_else(|| "truncated u64 field".to_string())?;
    *offset = end;
    Ok(u64::from_le_bytes(chunk.try_into().unwrap()))
}

fn read_f64(bytes: &[u8], offset: &mut usize) -> Result<f64, String> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| "input offset overflow".to_string())?;
    let chunk = bytes
        .get(*offset..end)
        .ok_or_else(|| "truncated f64 field".to_string())?;
    *offset = end;
    Ok(f64::from_le_bytes(chunk.try_into().unwrap()))
}

fn parse_common(bytes: &[u8]) -> Result<(u8, usize, usize, usize, usize, usize), String> {
    if bytes.get(..8) != Some(MAGIC.as_slice()) {
        return Err("invalid input magic".to_string());
    }
    let dtype = *bytes.get(8).ok_or_else(|| "missing dtype".to_string())?;
    let mut offset = 16;
    let rows = usize::try_from(read_u64(bytes, &mut offset)?)
        .map_err(|_| "row count does not fit usize".to_string())?;
    let columns = usize::try_from(read_u64(bytes, &mut offset)?)
        .map_err(|_| "column count does not fit usize".to_string())?;
    let selections = usize::try_from(read_u64(bytes, &mut offset)?)
        .map_err(|_| "selection count does not fit usize".to_string())?;
    let seed_index = usize::try_from(read_u64(bytes, &mut offset)?)
        .map_err(|_| "seed index does not fit usize".to_string())?;
    if rows == 0 || columns == 0 || selections == 0 || selections > rows {
        return Err("invalid input dimensions".to_string());
    }
    if seed_index >= rows {
        return Err("seed index is outside the input".to_string());
    }
    Ok((dtype, rows, columns, selections, seed_index, offset))
}

fn parse_prefix(
    bytes: &[u8],
    columns: usize,
    mut offset: usize,
) -> Result<(Vec<bool>, Vec<f64>, usize), String> {
    let mask_end = offset
        .checked_add(columns)
        .ok_or_else(|| "directional mask overflow".to_string())?;
    let directional = bytes
        .get(offset..mask_end)
        .ok_or_else(|| "truncated directional mask".to_string())?
        .iter()
        .map(|value| *value != 0)
        .collect();
    offset = mask_end;
    let mut deviations = Vec::with_capacity(columns);
    for _ in 0..columns {
        let deviation = read_f64(bytes, &mut offset)?;
        if !deviation.is_finite() || deviation <= 0.0 {
            return Err("invalid deviation".to_string());
        }
        deviations.push(deviation);
    }
    Ok((directional, deviations, offset))
}

fn parse_f64(bytes: &[u8]) -> Result<Input<f64>, String> {
    let (dtype, rows, columns, selections, seed_index, offset) = parse_common(bytes)?;
    if dtype != 8 {
        return Err("expected float64 input".to_string());
    }
    let (directional, deviations, mut offset) = parse_prefix(bytes, columns, offset)?;
    let value_count = rows
        .checked_mul(columns)
        .ok_or_else(|| "data size overflow".to_string())?;
    let mut data = Vec::with_capacity(value_count);
    for _ in 0..value_count {
        data.push(read_f64(bytes, &mut offset)?);
    }
    if offset != bytes.len() {
        return Err("unexpected trailing bytes".to_string());
    }
    Ok(Input {
        rows,
        columns,
        selections,
        seed_index,
        directional,
        deviations,
        data,
    })
}

fn parse_f32(bytes: &[u8]) -> Result<Input<f32>, String> {
    let (dtype, rows, columns, selections, seed_index, offset) = parse_common(bytes)?;
    if dtype != 4 {
        return Err("expected float32 input".to_string());
    }
    let (directional, deviations, mut offset) = parse_prefix(bytes, columns, offset)?;
    let value_count = rows
        .checked_mul(columns)
        .ok_or_else(|| "data size overflow".to_string())?;
    let mut data = Vec::with_capacity(value_count);
    for _ in 0..value_count {
        let end = offset
            .checked_add(4)
            .ok_or_else(|| "input offset overflow".to_string())?;
        let chunk = bytes
            .get(offset..end)
            .ok_or_else(|| "truncated f32 field".to_string())?;
        offset = end;
        data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    if offset != bytes.len() {
        return Err("unexpected trailing bytes".to_string());
    }
    Ok(Input {
        rows,
        columns,
        selections,
        seed_index,
        directional,
        deviations,
        data,
    })
}

fn select<T: DataValue>(input: &Input<T>) -> Vec<usize> {
    let mut indices = Vec::with_capacity(input.selections);
    indices.push(input.seed_index);
    if input.selections == 1 {
        return indices;
    }

    let mut selected = vec![false; input.rows];
    selected[input.seed_index] = true;
    let mut nearest = vec![f64::INFINITY; input.rows];

    while indices.len() < input.selections {
        let last_index = *indices.last().unwrap();
        let last_offset = last_index * input.columns;
        let last_row = &input.data[last_offset..last_offset + input.columns];
        for (row_index, row) in input.data.chunks_exact(input.columns).enumerate() {
            let mut distance = 0.0_f64;
            for (((&left, &right), &deviation), &directional) in row
                .iter()
                .zip(last_row)
                .zip(&input.deviations)
                .zip(&input.directional)
            {
                let mut difference = T::difference(left, right);
                if directional {
                    difference = (difference + 180.0).rem_euclid(360.0) - 180.0;
                }
                let normalized = difference / deviation;
                distance += normalized * normalized;
            }
            if distance < nearest[row_index] {
                nearest[row_index] = distance;
            }
        }

        let mut next_index = 0;
        let mut greatest_distance = f64::NEG_INFINITY;
        for (row_index, (nearest_value, &is_selected)) in
            nearest.iter_mut().zip(&selected).enumerate()
        {
            if is_selected {
                *nearest_value = f64::NEG_INFINITY;
            } else if *nearest_value > greatest_distance {
                greatest_distance = *nearest_value;
                next_index = row_index;
            }
        }
        indices.push(next_index);
        selected[next_index] = true;
    }
    indices
}

fn format_usize(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn format_f64(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.12}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn run() -> Result<(), String> {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len() != 3 {
        return Err("usage: rust_candidate INPUT REPEATS".to_string());
    }
    let repeats = arguments[2]
        .parse::<usize>()
        .map_err(|_| "REPEATS must be a positive integer".to_string())?;
    if repeats == 0 {
        return Err("REPEATS must be positive".to_string());
    }

    let read_started = Instant::now();
    let bytes = fs::read(&arguments[1]).map_err(|error| error.to_string())?;
    let dtype = *bytes.get(8).ok_or_else(|| "missing dtype".to_string())?;

    let mut timings = Vec::with_capacity(repeats);
    if dtype == 8 {
        let input = parse_f64(&bytes)?;
        let input_read_s = read_started.elapsed().as_secs_f64();
        let mut final_indices = Vec::new();
        for _ in 0..repeats {
            let started = Instant::now();
            final_indices = select(&input);
            timings.push(started.elapsed().as_secs_f64());
        }
        println!(
            "{{\"input_read_s\":{input_read_s:.12},\"selection_s\":[{}],\"indices\":[{}]}}",
            format_f64(&timings),
            format_usize(&final_indices),
        );
        return Ok(());
    }
    if dtype == 4 {
        let input = parse_f32(&bytes)?;
        let input_read_s = read_started.elapsed().as_secs_f64();
        let mut final_indices = Vec::new();
        for _ in 0..repeats {
            let started = Instant::now();
            final_indices = select(&input);
            timings.push(started.elapsed().as_secs_f64());
        }
        println!(
            "{{\"input_read_s\":{input_read_s:.12},\"selection_s\":[{}],\"indices\":[{}]}}",
            format_f64(&timings),
            format_usize(&final_indices),
        );
        return Ok(());
    }
    Err(format!("unsupported dtype marker: {dtype}"))
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(2);
    }
}
