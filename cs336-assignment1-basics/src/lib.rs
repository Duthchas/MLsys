use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::HashMap;

type Token = Vec<Box<[u8]>>;
type Pair = (Box<[u8]>, Box<[u8]>);

#[pyfunction]
fn train_bpe_rs<'py>(
    py: Python<'py>,
    pre_token_counts_py: &Bound<'py, PyList>,
    vocab_size: usize,
    special_tokens_py: &Bound<'py, PyList>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyList>)> {
    // Initialize vocabulary
    let mut vocab: Vec<Box<[u8]>> = Vec::with_capacity(vocab_size);
    for i in 0..256usize {
        vocab.push(Box::new([i as u8]));
    }
    let mut next_id: usize = 256;

    for item in special_tokens_py.iter() {
        let st_bytes: &[u8] = item.downcast::<PyBytes>()?.as_bytes();
        vocab.push(st_bytes.into());
        next_id += 1;
    }

    let num_merges = vocab_size - next_id;

    // Parse pre_token_counts: list of (bytes, int)
    // bytes is the raw token bytes, we split into individual bytes for BPE
    let mut pre_token_counts: HashMap<Token, u32> = HashMap::new();
    for item in pre_token_counts_py.iter() {
        let tuple = item.downcast::<PyTuple>()?;
        let py_bytes = tuple.get_item(0)?.downcast::<PyBytes>()?.clone();
        let token_bytes: &[u8] = py_bytes.as_bytes();
        let freq: u32 = tuple.get_item(1)?.extract()?;

        let token: Token = token_bytes.iter().map(|&b| Box::new([b]) as Box<[u8]>).collect();
        *pre_token_counts.entry(token).or_insert(0) += freq;
    }

    if pre_token_counts.is_empty() {
        let vocab_py = PyDict::new(py);
        for (i, v) in vocab.iter().enumerate() {
            vocab_py.set_item(i, PyBytes::new(py, v))?;
        }
        return Ok((vocab_py, PyList::empty(py)));
    }

    // Build pair -> {token -> weighted_count} index AND cached pair totals
    let mut pair_in_token: HashMap<Pair, HashMap<Token, u32>> = HashMap::new();
    let mut pair_total_count: HashMap<Pair, u32> = HashMap::new();

    for (token, &freq) in &pre_token_counts {
        for i in 0..token.len() - 1 {
            let pair = (token[i].clone(), token[i + 1].clone());
            *pair_total_count.entry(pair.clone()).or_insert(0) += freq;
            *pair_in_token
                .entry(pair)
                .or_default()
                .entry(token.clone())
                .or_insert(0) += freq;
        }
    }

    let mut merges: Vec<(Box<[u8]>, Box<[u8]>)> = Vec::with_capacity(num_merges);

    for _ in 0..num_merges {
        // Find most frequent pair from cached totals (no recomputing sums)
        let best_pair: Pair = match pair_total_count
            .iter()
            .max_by(|(pa, &ca), (pb, &cb)| {
                ca.cmp(&cb).then_with(|| pa.cmp(pb))
            }) {
            Some((pair, &count)) if count > 0 => pair.clone(),
            _ => break,
        };

        pair_total_count.remove(&best_pair);

        let merged: Box<[u8]> = [best_pair.0.as_ref(), best_pair.1.as_ref()].concat().into();
        merges.push((best_pair.0.clone(), best_pair.1.clone()));
        vocab.push(merged.clone());
        next_id += 1;

        // Snapshot affected tokens
        let affected: Vec<(Token, u32)> = match pair_in_token.remove(&best_pair) {
            Some(tokens) => tokens.into_iter().collect(),
            None => continue,
        };

        // Decrement pair_total_count for old pairs of affected tokens
        for (old_token, _) in &affected {
            for i in 0..old_token.len() - 1 {
                let pair = (old_token[i].clone(), old_token[i + 1].clone());
                if pair == best_pair {
                    continue;
                }
                if let Some(tokens) = pair_in_token.get_mut(&pair) {
                    if let Some(freq) = tokens.remove(old_token) {
                        let total = pair_total_count.entry(pair.clone()).or_insert(0);
                        *total = total.saturating_sub(freq);
                        if *total == 0 {
                            pair_total_count.remove(&pair);
                        }
                    }
                    if tokens.is_empty() {
                        pair_in_token.remove(&pair);
                    }
                }
            }
        }

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

            // Add new token's pairs to pair_in_token and pair_total_count
            for i in 0..new_token.len() - 1 {
                let pair = (new_token[i].clone(), new_token[i + 1].clone());
                *pair_total_count.entry(pair.clone()).or_insert(0) += freq;
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
