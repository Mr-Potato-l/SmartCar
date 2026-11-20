#![allow(dead_code)]
mod neural;
mod car;

use std::time::Instant;
use macroquad::prelude::*;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use ndarray::array;
use neural::NeuralNetwork;
use car::{Car, Controls};

fn create_network() -> NeuralNetwork {
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

    let targets = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];

    let hidden = 8;

    let learning_rate = 0.05;

    let network = NeuralNetwork::new(input_vectors, targets, hidden, learning_rate);

    return network;
}

#[macroquad::main("Car Game")]
async fn main() {
    let mut car = Car { x: 100.0, y: 100.0, angle: 0.0, speed: 0.0, alive: true };
    let center_x = 400.0;
    let center_y = 400.0;
    let inner_ring_r = 170.0;
    let outer_ring_r = 300.0;

    loop {
        let mut controls = Controls { steering: 0.0, throttle: 0.0 };

        if is_key_down(KeyCode::W) {
            controls.throttle = 1.0;
        } else if is_key_down(KeyCode::S) {
            controls.throttle = -1.0;
        }

        if is_key_down(KeyCode::A) {
            controls.steering = 1.0;
        } else if is_key_down(KeyCode::D) {
            controls.steering = -1.0;
        }

        car.update(&controls, get_frame_time());
        
        clear_background(RED);

        draw_circle(center_x, center_y, outer_ring_r, BLACK);

        draw_circle(center_x, center_y, inner_ring_r, RED);

        let dist = (car.x - center_x).hypot(car.y - center_y);

        let car_color = if dist <= outer_ring_r - 10.0 && dist >= inner_ring_r + 10.0 {
            BLUE
        } else { GREEN };

        draw_line(car.x, car.y, car.get_x_at_len(), car.get_y_at_len(), 30.0, car_color);

        draw_text("Hello Macroquad!", 20.0, 20.0, 30.0, DARKGRAY);


        next_frame().await
    }
}