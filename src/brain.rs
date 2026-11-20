#![allow(dead_code)]

use ndarray::{Array1, Array2, ArrayView1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::rand::Rng;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn leaky_relu(x: f32) -> f32 {
    if x > 0.0 { return x } else { return x * 0.01 }
}

fn tanh(x: f32) -> f32 {
    x.sinh() / x.cosh()
}

#[derive(Clone)]
pub struct EvolvingNetwork {
    weights2: Array2<f32>,
    bias2: Array1<f32>,
    weights1: Array2<f32>,
    bias1: Array1<f32>,
}

impl EvolvingNetwork {
    pub fn new(input_len: usize, output_len: usize, hidden: usize) -> Self {
        let weights1 = Array2::random((input_len, hidden), StandardNormal) * 0.5;
        let bias1 = Array1::random(hidden, StandardNormal) * 0.1;
        let weights2 = Array2::random((hidden, output_len), StandardNormal) * 0.5;
        let bias2 = Array1::random(output_len, StandardNormal);

        Self { weights1, bias1, weights2, bias2 }
    }

    pub fn predict(&self, input: ArrayView1<f32>) -> Array1<f32> {
        let layer1 = input.dot(&self.weights1) + &self.bias1;
        let layer2 = layer1.mapv(leaky_relu);
        let layer3 = layer2.dot(&self.weights2) + &self.bias2;
        layer3.mapv(tanh)
    }

    pub fn mutate(&mut self, rng: &mut impl Rng) {
        self.weights1.mapv_inplace(|x| x + rng.sample::<f32, _>(StandardNormal));
        self.bias1.mapv_inplace(|x| x + rng.sample::<f32, _>(StandardNormal));
        self.weights2.mapv_inplace(|x| x + rng.sample::<f32, _>(StandardNormal));
        self.bias2.mapv_inplace(|x| x + rng.sample::<f32, _>(StandardNormal));
    }
}
