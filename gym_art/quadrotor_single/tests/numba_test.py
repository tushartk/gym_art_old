import unittest
import time
from gym_art.quadrotor_multi.numba_utils import OUNoiseNumba
from gym_art.quadrotor_multi.quad_utils import OUNoise
from gym_art.quadrotor_single.sensor_noise import SensorNoise
import numpy

def create_env(use_numba=False):
    from gym_art.quadrotor_single.quadrotor import QuadrotorEnv

    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None

    episode_duration = 7  # seconds

    raw_control = raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    sense_noise = 'default'


    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    env = QuadrotorEnv(
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=True, ep_time=episode_duration,
        use_numba=use_numba
    )

    return env


class TestNumbaOpt(unittest.TestCase):
    def test_optimized_env(self):
        env = create_env(use_numba=True)

        env.reset()
        time.sleep(0.1)

        num_steps = 0
        while num_steps < 100:
            obs, rewards, dones, infos = env.step(env.action_space.sample())
            num_steps += 1
            print('Rewards: ', rewards)

        env.close()

    @staticmethod
    def step_env(use_numba, steps):
        env = create_env(use_numba=use_numba)
        env.reset()
        num_steps = 0

        # warmup
        for i in range(20):
            obs, rewards, dones, infos = env.step(env.action_space.sample())
            num_steps += 1

        print('Measuring time, numba:', use_numba)
        start = time.time()
        for i in range(steps):
            obs, rewards, dones, infos = env.step(env.action_space.sample())
            if dones:
                env.reset()

        elapsed_sec = time.time() - start
        fps = steps / elapsed_sec
        return fps, elapsed_sec

    def test_performance_difference(self):
        steps = 1000
        fps, elapsed_sec = self.step_env(use_numba=False, steps=steps)
        fps_numba, elapsed_sec_numba = self.step_env(use_numba=True, steps=steps)

        print('Regular: ', fps, elapsed_sec)
        print('Numba: ', fps_numba, elapsed_sec_numba)

    def test_step_and_noise_opt(self):
        for _ in range(30):
            env = create_env()
            env.reset()

            dynamics = env.dynamics

            dt = 0.005
            thrust_noise_ratio = 0.05
            thrusts = numpy.random.random(4)

            import copy
            dynamics_copy = copy.deepcopy(dynamics)
            dynamics_copy_numba = copy.deepcopy(dynamics)

            dynamics.thrust_noise = OUNoise(4, sigma=0.2 * thrust_noise_ratio, use_seed=True)
            dynamics_copy_numba.thrust_noise = OUNoiseNumba(4, sigma=0.2 * thrust_noise_ratio, use_seed=True)
            thrust_noise = thrust_noise_copy = dynamics.thrust_noise.noise()
            thrust_noise_numba = dynamics_copy_numba.thrust_noise.noise()
            dynamics.step1(thrusts, dt, thrust_noise)
            dynamics_copy.step1(thrusts, dt, thrust_noise_copy)
            dynamics_copy_numba.step1_numba(thrusts, dt, thrust_noise_numba)

            def pos_vel_acc_tor(d):
                return d.pos, d.vel, d.acc, d.torque

            def rot_omega_accm(d):
                return d.rot, d.omega, d.accelerometer

            p1, v1, a1, t1 = pos_vel_acc_tor(dynamics)
            p2, v2, a2, t2 = pos_vel_acc_tor(dynamics_copy)
            p3, v3, a3, t3 = pos_vel_acc_tor(dynamics_copy_numba)

            self.assertTrue(numpy.allclose(p1, p2))
            self.assertTrue(numpy.allclose(v1, v2))
            self.assertTrue(numpy.allclose(a1, a2))
            self.assertTrue(numpy.allclose(t1, t2))

            self.assertTrue(numpy.allclose(p1, p3))
            self.assertTrue(numpy.allclose(v1, v3))
            self.assertTrue(numpy.allclose(a1, a3))
            self.assertTrue(numpy.allclose(t1, t3))

            # the below test is to check if add_noise is returning the same value
            r1, o1, accm1 = rot_omega_accm(dynamics)
            r2, o2, accm2 = rot_omega_accm(dynamics_copy_numba)

            sense_noise = SensorNoise(bypass=False)
            sense_noise_numba = SensorNoise(bypass=False)

            new_p1, new_v1, new_r1, new_o1, new_a1 = sense_noise.add_noise(p1, v1, r1, o1, accm1, dt)
            new_p2, new_v2, new_r2, new_o2, new_a2 = sense_noise_numba.add_noise_numba(p2, v2, r2, o2, accm2, dt)

            self.assertTrue(numpy.allclose(new_p1, new_p2))
            self.assertTrue(numpy.allclose(new_v1, new_v2))
            self.assertTrue(numpy.allclose(new_a1, new_a2))
            self.assertTrue(numpy.allclose(new_o1, new_o2))
            self.assertTrue(numpy.allclose(new_r1, new_r2))
            env.close()


if __name__ == '__main__':
    unittest.main()
