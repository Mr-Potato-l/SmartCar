#![allow(dead_code)]

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use std::time::Instant;
use ndarray_rand::rand::seq::SliceRandom;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64{
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn leaky_relu(x: f64) -> f64 {
    if x > 0.0 { return x } else { return x * 0.01 }
}

fn leaky_relu_derivative(x: f64) -> f64 {
    if x > 0.0 { return 1.0 } else { return 0.01 }
}

fn mse(prediction: f64, target: f64) -> f64 {
    (prediction - target).powf(2.0)
}

fn mse_derivative(prediction: f64, target: f64) -> f64 {
    2.0 * (prediction - target)
}

fn bce(prediction: f64, target: f64) -> f64 {
    let p = prediction.clamp(1e-12, 1.0 - 1e-12);
    -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
}

struct NeuralNetwork {
    weights2: Array1<f64>,
    bias2: f64,
    weights1: Array2<f64>,
    bias1: Array1<f64>,
    learning: f64,
}

impl NeuralNetwork {
    fn new(learning: f64, input_len: usize, input_hidden: usize) -> Self {
        let weights1 = Array2::random((input_len, input_hidden), StandardNormal) * 0.5;
        let bias1 = Array1::random(input_hidden, StandardNormal) * 0.1;
        let weights2 = Array1::random(input_hidden, StandardNormal) * 0.5;
        let bias2 = 0.0;

        Self { weights1, bias1, weights2, bias2, learning }
    }

    fn predict(&self, input: ArrayView1<f64>) -> f64 {
        let layer1 = input.dot(&self.weights1) + &self.bias1;
        let layer2 = layer1.mapv(leaky_relu);
        let layer3 = layer2.dot(&self.weights2) + &self.bias2;
        return sigmoid(layer3);
    }

    fn compute_gradients(&self, input: ArrayView1<f64>, target: f64) -> (Array2<f64>, Array1<f64>, Array1<f64>, f64) {
        let layer1 = input.dot(&self.weights1) + &self.bias1;
        let layer2 = layer1.mapv(leaky_relu);
        let layer3 = layer2.dot(&self.weights2) + &self.bias2;
        let prediction = sigmoid(layer3);

        let db2 = prediction - target;
        let dw2 = db2 * &layer2;
        let db1 = (db2 * &self.weights2) * layer1.mapv(leaky_relu_derivative);
        let dw1 = input.insert_axis(Axis(1)).dot(&db1.view().insert_axis(Axis(0)));
        
        return (dw1, db1, dw2, db2);
    }

    fn update_parameters(&mut self, dw1: ArrayView2<f64>, db1: ArrayView1<f64>, dw2: ArrayView1<f64>, db2: f64) {
        self.weights1 -= &(&dw1 * self.learning);
        self.bias1 -= &(&db1 * self.learning);
        self.weights2 -= &(&dw2 * self.learning);
        self.bias2 -= db2 * self.learning;
    }

    fn train(&mut self, input_vectors: ArrayView2<f64>, targets: ArrayView1<f64>, iterations: usize) -> Vec<f64> {
        let mut errors: Vec<f64> = vec![];
        let mut rng = ndarray_rand::rand::rng();
        
        let mut indices: Vec<usize> = (0..input_vectors.nrows()).collect();
        
        for iter in 0..iterations {
            indices.shuffle(&mut rng);

            for &index in &indices {
                let input = input_vectors.row(index);
                let target = targets[index];

                let (dw1, db1, dw2, db2) = self.compute_gradients(input, target);
                self.update_parameters(dw1.view(), db1.view(), dw2.view(), db2);
            }
            

            if iter % 100 == 0 {
                let mut cur_error = 0.0;
                for data_index in 0..input_vectors.nrows() {
                    let data_point = input_vectors.row(data_index);
                    let target = targets[data_index];

                    let prediction = self.predict(data_point);
                    let error = bce(prediction, target);

                    cur_error += error;
                }
                errors.push(cur_error);
            }
        }

        return errors;
    }
}

fn main() {
    let input_vectors = array![
        [2.0, 1.5],
        [2.0, 1.0],
        [1.0, 1.5],
        [1.5, 2.0],
        [2.5, 0.5],
        [2.0, 0.5],
        [2.5, 1.0],
        [1.0, 1.0],
    ];

    let mean= input_vectors.mean_axis(Axis(0)).unwrap();

    let input_vectors = &input_vectors - &mean;

    let targets = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];

    let hidden = 8;

    let iterations = 2000;

    let learning_rate = 0.05;

    let mut network = NeuralNetwork::new(learning_rate, input_vectors.ncols(), hidden);

    let start = Instant::now();

    let errors = network.train(input_vectors.view(), targets.view(), iterations);

    let duration = start.elapsed();

    let mut sum = 0.0;

    for (step, error) in errors.iter().enumerate() {
        println!("Step: {}, Error: {}", step + 1, error);
        sum += error;
    }

    println!("Finished {} iterations in {:?}", iterations, duration);
    println!("Final Error: {}, Avg. Error: {}", errors.last().unwrap(), (sum / errors.len().to_f64().unwrap()));
    
}