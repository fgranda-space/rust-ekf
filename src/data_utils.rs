// src/data_utils.rs
use polars::datatypes::PlSmallStr;
use polars::prelude::*;
use std::error::Error;
use std::fs::File;
use std::path::Path;

pub fn read_space_separated_file(
    path: &str,
    headers: &[&str],
    n_rows: Option<usize>,
) -> PolarsResult<DataFrame> {
    let file = File::open(path)?;

    let parse_options = CsvParseOptions::default().with_separator(b' ');

    let mut read_options = CsvReadOptions::default()
        .with_has_header(false)
        .with_parse_options(parse_options);

    // Set n_rows if specified
    if let Some(rows) = n_rows {
        read_options = read_options.with_n_rows(Some(rows));
    }

    let mut df = CsvReader::new(file).with_options(read_options).finish()?;

    let old_names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    for (i, &new_name) in headers.iter().enumerate() {
        df.rename(&old_names[i], PlSmallStr::from(new_name))?;
    }
    Ok(df)
}

/// Saves a DataFrame to a CSV file.
pub fn save_csv<P: AsRef<Path>>(df: &mut DataFrame, path: P) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    CsvWriter::new(file).include_header(true).finish(df)?;
    Ok(())
}
