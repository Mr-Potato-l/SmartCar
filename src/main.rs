#![allow(dead_code)]
mod neural;
mod car;

use macroquad::prelude::*;
use ndarray_rand::rand_distr::num_traits::Float;
use ndarray::{Array1, Array2, array};
use neural::NeuralNetwork;
use car::{Car, Controls};
use std::vec::Vec;

const CENTER_X: f32 = 240.0;
const CENTER_Y: f32 = 400.0;
const INNER_RADIUS: f32 = 110.0;
const OUTER_RADIUS: f32 = 230.0;
const STEP_SIZE: f32 = 2.0;
const RECORDING_CAP: usize = 1000;
const RESTART_X: f32 = 70.0;
const RAY_RANGE: f32 = 60.0;

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

fn create_driving_network(inputs: Array2<f32>, targets: Array1<f32>) -> NeuralNetwork {
    let hidden = 8;

    let learning_rate = 0.05;

    let network = NeuralNetwork::new(inputs, targets, hidden, learning_rate);

    return network;
}

fn train_network(network: &mut NeuralNetwork) -> Vec<f32> {
    network.train(100)
}

fn is_on_road(x: f32, y: f32) -> bool {
    let closest = x.clamp(CENTER_X, CENTER_X + 2.0 * OUTER_RADIUS);
    let dist = (x - closest).hypot(y - CENTER_Y);
    dist <= OUTER_RADIUS - 4.0 && dist >= INNER_RADIUS + 4.0
}

fn raycast(x: f32, y: f32, angle: f32, max_range: f32) -> f32 {
    let mut steps = 0;
    let mut px = x;
    let mut py = y;
    let angle_cos = angle.cos();
    let angle_sin = angle.sin();

    while is_on_road(px, py) && steps as f32 * STEP_SIZE < max_range {
        steps += 1;

        px += STEP_SIZE * angle_cos;
        py += STEP_SIZE * angle_sin;
    }

    draw_line(x, y, px, py, 4.0, GRAY);

    ((steps as f32 * STEP_SIZE) / max_range).clamp(0.0, 1.0)
}

fn get_rays(car: &Car) -> Vec<f32> {
    let ray_angles = [-45.0, 0.0, 45.0];
    let mut rays: Vec<f32> = vec![];

    for angle in ray_angles {
        rays.push(raycast(car.x, car.y, car.angle + angle.to_radians(), RAY_RANGE));
    }

    return rays;
}

fn center_steering(steering: f32) -> f32 {
    (steering + 1.0) / 2.0
}

fn reset_car(car: &mut Car) {
    car.x = RESTART_X;
    car.y = CENTER_Y;
    car.angle = (-90.0).to_radians();
    car.speed = 0.0;
    car.alive = true;
}

#[macroquad::main("Car Game")]
async fn main() {
    let mut car = Car { x: RESTART_X, y: CENTER_Y, angle: (-90.0).to_radians(), speed: 0.0, alive: true };
    let mut rays_recording: Vec<Vec<f32>> = vec![];
    let mut steering_recording: Vec<f32> = vec![];
    let mut count = 0;
    let mut frame = 0;
    let mut neural_mode = false;
    let mut network: neural::NeuralNetwork = create_network();
    let mut steer = 0.0;
    let mut error: f32 = 0.0;

    loop {
        let mut controls = Controls { steering: 0.0, throttle: 0.5 };

        if !neural_mode {
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
        } else {
            controls.steering = steer;
        }

        if is_key_pressed(KeyCode::R) || !car.alive {
            reset_car(&mut car);
            if neural_mode {
                error = *train_network(&mut network).last().unwrap();
            }
        }

        if is_key_pressed(KeyCode::M) {
            reset_car(&mut car);
            neural_mode = false;
        }

        if is_key_pressed(KeyCode::N) && count == RECORDING_CAP && !neural_mode {
            neural_mode = true;
            
            let rows = rays_recording.len();
            let cols = rays_recording.first().map_or(0, Vec::len);
            let rays = rays_recording.clone().into_iter().flatten().collect::<Vec<f32>>();
            network = create_driving_network(Array2::from_shape_vec((rows, cols), rays).unwrap(), Array1::from_vec(steering_recording.clone()));
            error = *train_network(&mut network).last().unwrap();
            reset_car(&mut car);
        }

        if is_key_pressed(KeyCode::T) && neural_mode {
            error = *train_network(&mut network).last().unwrap();
            reset_car(&mut car);
        }

        car.update(&controls, get_frame_time());
        
        clear_background(RED);

        draw_circle(CENTER_X, CENTER_Y, OUTER_RADIUS, BLACK);
        draw_circle(CENTER_X + 2.0 * OUTER_RADIUS, CENTER_Y, OUTER_RADIUS, BLACK);
        draw_rectangle(CENTER_X, CENTER_Y - OUTER_RADIUS, OUTER_RADIUS * 2.0, OUTER_RADIUS * 2.0, BLACK);

        draw_circle(CENTER_X, CENTER_Y, INNER_RADIUS, RED);
        draw_circle(CENTER_X + 2.0 * OUTER_RADIUS, CENTER_Y, INNER_RADIUS, RED);
        draw_rectangle(CENTER_X, CENTER_Y - INNER_RADIUS, OUTER_RADIUS * 2.0, INNER_RADIUS * 2.0, RED);

        let car_color = if is_on_road(car.x, car.y) {
            car.alive = true;
            BLUE
        } else { car.alive = false; GREEN };

        draw_line(car.x, car.y, car.front_x(), car.front_y(), 30.0, car_color);

        draw_text(format!("Steering: {}", center_steering(controls.steering)), 20.0, 20.0, 30.0, DARKGRAY);

        draw_text(format!("Last Error: {}", error), 20.0, 100.0, 30.0, DARKGRAY);

        let rays = get_rays(&car);
        if car.alive && car.speed >= 20.0 && count < RECORDING_CAP && frame % 3 == 0 {
            rays_recording.push(rays.clone());
            steering_recording.push(center_steering(controls.steering));
            count += 1;
        }

        if count == RECORDING_CAP {
            let cr = steering_recording.iter().filter(|s| **s == 0.0).count();
            let cn = steering_recording.iter().filter(|s| **s == 0.5).count();
            let cl = steering_recording.iter().filter(|s| **s == 1.0).count();
            draw_text(format!("Right: {}, None: {}, Left: {}", cr, cn, cl), 20.0, 60.0, 30.0, DARKGRAY);
        }

        if neural_mode {
            steer = network.predict(Array1::from_vec(rays).view());
            steer = (steer * 2.0) - 1.0;
        }

        frame += 1;
        next_frame().await
    }
}