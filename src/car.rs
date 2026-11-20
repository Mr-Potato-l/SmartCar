#![allow(dead_code)]

use macroquad::prelude::*;

const MAX_SPEED: f32 = 70.0;
const MAX_REVERSE: f32 = -MAX_SPEED / 4.0;
const TURN_POWER: f32 = MAX_SPEED / 3.0;
const ACCELERATION: f32 = MAX_SPEED / 5.0;
const BRAKE: f32 = ACCELERATION * 4.0;
const DRAG: f32 = ACCELERATION / 2.0;
const SENSITIVITY: f32 = 1.3;
const LEN: f32 = 50.0;
const DRAG_DEADZONE: f32 = 0.05;

pub struct Controls {
    pub steering: f32,
    pub throttle: f32,
}

pub struct Car {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub speed: f32,
    pub alive: bool,
}

impl Car {
    pub fn update(&mut self, controls: &Controls, dtime: f32) {
        let throttle = if controls.throttle.abs() <= DRAG_DEADZONE { 0.0 } else {controls.throttle};
        self.angle -= controls.steering * (self.speed / TURN_POWER).clamp(-1.0, 1.0) * SENSITIVITY * dtime;

        let mut acceleration = ACCELERATION;
        if throttle * self.speed.signum() < 0.0 { acceleration = BRAKE; }
        self.speed += throttle * acceleration * dtime;
        if throttle == 0.0 { self.speed -= self.speed.signum() * (DRAG * dtime).min(self.speed.abs()); }
        self.speed = self.speed.clamp(MAX_REVERSE, MAX_SPEED);

        self.x += self.angle.cos() * self.speed * dtime;
        self.y += self.angle.sin() * self.speed * dtime;
    }

    pub fn front_x(&self) -> f32 {
        self.x + self.angle.cos() * -LEN
    }

    pub fn front_y(&self) -> f32 {
        self.y + self.angle.sin() * -LEN
    }
}