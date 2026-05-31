use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::HashMap;

type Token = Vec<Vec<u8>>;
type Pair = (Vec<u8>, Vec<u8>);

fn token_to_py<'py>(py: Python<'py>, token: &Token) -> PyResult<Bound<'py, PyTuple>> {
    let items: Vec<Bound<'_, PyBytes>> = token.iter().map(|b| PyBytes::new(py, b)).collect();
    PyTuple::new(py, &items)
}

fn pair_key(p: &[u8]) -> &[u8] {
    p
}

/// Full BPE training in Rust.
#[pyfunction]
fn train_bpe_rs<'py>(
    py: Python<'py>,
    pre_token_counts_py: &Bound<'py, PyList>,
    vocab_size: usize,
    special_tokens_py: &Bound<'py, PyList>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyList>)> {
    // Initialize vocabulary
    let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(vocab_size);
    for i in 0..256usize {
        vocab.push(vec![i as u8]);
    }
    let mut next_id: usize = 256;

    for item in special_tokens_py.iter() {
        let st_bytes: &[u8] = item.downcast::<PyBytes>()?.as_bytes();
        vocab.push(st_bytes.to_vec());
        next_id += 1;
    }

    let num_merges = vocab_size - next_id;

    // Parse pre_token_counts: list of (tuple_of_bytes, int)
    let mut pre_token_counts: HashMap<Token, u32> = HashMap::new();
    for item in pre_token_counts_py.iter() {
        let tuple = item.downcast::<PyTuple>()?;
        let token_tuple = tuple.get_item(0)?.downcast::<PyTuple>()?.clone();
        let freq: u32 = tuple.get_item(1)?.extract()?;

        let mut token: Token = Vec::with_capacity(token_tuple.len());
        for byte_item in token_tuple.iter() {
            let b: &[u8] = byte_item.downcast::<PyBytes>()?.as_bytes();
            token.push(b.to_vec());
        }
        *pre_token_counts.entry(token).or_insert(0) += freq;
    }

    if pre_token_counts.is_empty() {
        let vocab_py = PyDict::new(py);
        for (i, v) in vocab.iter().enumerate() {
            vocab_py.set_item(i, PyBytes::new(py, v))?;
        }
        return Ok((vocab_py, PyList::empty(py)));
    }

    // Build pair -> {token -> weighted_count} index
    let mut pair_in_token: HashMap<Pair, HashMap<Token, u32>> = HashMap::new();

    for (token, &freq) in &pre_token_counts {
        for i in 0..token.len() - 1 {
            let pair = (token[i].clone(), token[i + 1].clone());
            *pair_in_token
                .entry(pair)
                .or_default()
                .entry(token.clone())
                .or_insert(0) += freq;
        }
    }

    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(num_merges);

    for _ in 0..num_merges {
        // Find most frequent pair, ties broken by (p[0], p[1]) lexicographic
        let best_pair: Pair = match pair_in_token
            .iter()
            .map(|(pair, tokens)| {
                let total: u32 = tokens.values().sum();
                (total, pair.0.clone(), pair.1.clone(), pair.clone())
            })
            .max_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2)))
        {
            Some((count, _, _, pair)) if count > 0 => pair,
            _ => break,
        };

        let merged: Vec<u8> = [best_pair.0.as_slice(), best_pair.1.as_slice()].concat();
        merges.push((best_pair.0.clone(), best_pair.1.clone()));
        vocab.push(merged.clone());
        next_id += 1;

        // Snapshot affected tokens
        let affected: Vec<(Token, u32)> = match pair_in_token.remove(&best_pair) {
            Some(tokens) => tokens.into_iter().collect(),
            None => continue,
        };

        for (old_token, _) in affected {
            let freq = match pre_token_counts.remove(&old_token) {
                Some(f) => f,
                None => continue,
            };

            // Build new token
            let mut new_token: Token = Vec::with_capacity(old_token.len());
            let mut i = 0;
            while i < old_token.len() {
                if i + 1 < old_token.len()
                    && old_token[i] == best_pair.0
                    && old_token[i + 1] == best_pair.1
                {
                    new_token.push(merged.clone());
                    i += 2;
                } else {
                    new_token.push(old_token[i].clone());
                    i += 1;
                }
            }

            // Update pre-token counts
            *pre_token_counts.entry(new_token.clone()).or_insert(0) += freq;

            // Update pair_in_token: remove old token's pairs, add new token's pairs
            for i in 0..old_token.len() - 1 {
                let pair = (old_token[i].clone(), old_token[i + 1].clone());
                if pair == best_pair {
                    continue;
                }
                if let Some(tokens) = pair_in_token.get_mut(&pair) {
                    tokens.remove(&old_token);
                    if tokens.is_empty() {
                        pair_in_token.remove(&pair);
                    }
                }
            }

            for i in 0..new_token.len() - 1 {
                let pair = (new_token[i].clone(), new_token[i + 1].clone());
                *pair_in_token
                    .entry(pair)
                    .or_default()
                    .entry(new_token.clone())
                    .or_insert(0) += freq;
            }
        }
    }

    // Convert to Python
    let vocab_py = PyDict::new(py);
    for (i, v) in vocab.iter().enumerate() {
        vocab_py.set_item(i, PyBytes::new(py, v))?;
    }

    let merges_py = PyList::empty(py);
    for (b1, b2) in &merges {
        let t = PyTuple::new(py, &[PyBytes::new(py, b1), PyBytes::new(py, b2)])?;
        merges_py.append(t)?;
    }

    Ok((vocab_py, merges_py))
}

#[pymodule]
fn bpe_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe_rs, m)?)?;
    Ok(())
}
