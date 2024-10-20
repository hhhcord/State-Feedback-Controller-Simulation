import numpy as np
from ClassFiles.ControlSystemSimulation import ControlSystemSimulation
from ClassFiles.StateFeedbackControllerSimulation import StateFeedbackController

def main():
    # Initializing system parameters and performing simulation

    # Sampling time (Ts)
    sampling_time = 0.01

    # Time and step configuration
    steps = np.arange(0, 51)  # Step count from 0 to 50
    total_steps = len(steps)
    time_vector = np.arange(0, total_steps * sampling_time, sampling_time)  # Time vector

    # Rectangular wave signal input
    input_signal = np.zeros(total_steps)
    input_signal[2:7] = 1  # Create a rectangular pulse between time steps 2 and 6

    # System order for the simulation
    system_order = 2

    # Initialize control system simulation
    print("Initializing the control system simulation...")
    simulation = ControlSystemSimulation(n=system_order, t_end=total_steps * sampling_time, num_points=total_steps)

    # Define plant system matrices
    plant_A = np.array([[1, 1], [0, -2]])
    plant_B = np.array([[0], [1]])
    plant_C = np.array([1, 0])
    plant_D = np.array([0])
    initial_controller_gain = np.array([0.8, -2.0])
    F_ast = np.array([0.5, -1.5])

    # Build the plant digital system
    plant_system = simulation.build_digital_system(plant_A, plant_B, plant_C, plant_D)

    # Define ideal system matrices
    ideal_A = np.array([[1, 1], [-0.5, -0.5]])
    ideal_B = np.array([[0], [1]])
    ideal_C = np.array([1, 0])
    ideal_D = np.array([0])

    # Build the ideal digital system
    ideal_system = simulation.build_digital_system(ideal_A, ideal_B, ideal_C, ideal_D)

    # Set up the state feedback controller
    print("Setting up the state feedback controller...")
    state_feedback_controller = StateFeedbackController(
        n=system_order,
        plant_system=plant_system,
        ideal_system=ideal_system,
        input_signal=input_signal,
        test_signal=input_signal,  # Using the same signal for testing
        sampling_rate=1 / sampling_time,
        F_ini=initial_controller_gain, 
        F_ast=F_ast
    )

    # Run the simulation
    print("Running the state feedback controller simulation...")
    state_feedback_controller.run()
    print("Simulation completed.")

if __name__ == "__main__":
    main()
