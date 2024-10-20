import pandas as pd
import numpy as np
import ast
from ClassFiles.AudioLoader import AudioLoader  # Import the AudioLoader class
from ClassFiles.FrequencyResponseAnalyzer import FrequencyResponseAnalyzer  # Import the FrequencyResponseAnalyzer class
from ClassFiles.ControlSystemSimulation import ControlSystemSimulation
from ClassFiles.StateFeedbackControllerSimulation import StateFeedbackController

def generate_initial_controller_gain(system_order, lower_bound=-5e-2, upper_bound=5e-2):
    """
    Generates the initial controller gain based on the specified system order.
    
    Parameters:
    system_order (int): The size of the random array (system order)
    lower_bound (float): Lower bound of the random values (default is -1.0)
    upper_bound (float): Upper bound of the random values (default is 1.0)

    Returns:
    np.array: A random array representing the initial controller gain
    """
    # Generate random values within the specified range
    return np.random.uniform(lower_bound, upper_bound, size=system_order)

def load_state_feedback_gain(file_path='output/gain.csv'):
    """
    Reads the state feedback gain F from a CSV file and returns it as a NumPy array.

    Parameters:
    file_path (str): The path to the gain.csv file (default is 'output/gain.csv').

    Returns:
    np.array: State feedback gain F as a NumPy array.
    """
    # Read the CSV file
    gain_df = pd.read_csv(file_path)

    # Extract the 'Values' column from the 'State Feedback Gain (F)' row
    gain_values_str = gain_df[gain_df['Gain Type'] == 'State Feedback Gain (F)']['Values'].values[0]

    # Convert the string representation of the list to an actual list
    state_feedback_gain_list = ast.literal_eval(gain_values_str)

    # Convert the list to a NumPy array
    state_feedback_gain_array = np.array(state_feedback_gain_list)

    return state_feedback_gain_array

def main():
    # Create an instance of AudioLoader
    print("Creating AudioLoader instance...")
    al = AudioLoader()

    # Specify the duration in seconds to read
    time_test = 5  # Time duration for input/output audio data
    time_input = 27  # Time duration for test audio data

    # Load the input audio signal for a specified time period
    print("\nPlease select the .wav file for the input audio signal")
    input_data, sampling_rate = al.load_audio(time_test)
    print("Input audio signal loaded.")

    # Load the output audio signal for the same time period
    print("\nPlease select the .wav file for the output audio signal")
    output_data, _ = al.load_audio(time_test)
    print("Output audio signal loaded.")

    # Load the ideal output audio signal
    print("\nPlease select the .wav file for the ideal output audio signal")
    ideal_output_data, _ = al.load_audio(time_test)
    print("Ideal output audio signal loaded.")

    # Load the test audio signal for a different time period
    print("\nPlease select the .wav file for the test audio signal")
    test_data, _ = al.load_audio(time_input)
    print("Test audio signal loaded.")

    # Create an instance of FrequencyResponseAnalyzer and perform analysis
    print("Creating FrequencyResponseAnalyzer instance and analyzing frequency response...")
    fra = FrequencyResponseAnalyzer(input_signal=input_data, output_signal=output_data, sampling_rate=sampling_rate, time_duration=time_test)
    fra.analyze_and_save_bode_plot()
    print("Frequency response analysis completed and Bode plot saved.")

    # Specify the order of the system for simulation
    system_order = 175

    # Set up the control system simulation
    print("Setting up the control system simulation...")
    simulation = ControlSystemSimulation(n=system_order, t_end=time_test, num_points=len(input_data))

    # Plot the input and output signals
    print("Plotting input and output signals...")
    simulation.plot_input_output(input_data, output_data, filename='input_output_plot.png')
    simulation.plot_input_output(input_data, ideal_output_data, filename='input_ideal_output_plot.png')

    # Identify the system using SRIM method
    print("Identifying system using SRIM method...")
    SRIM_plant_system = simulation.identify_system_SRIM(input_data, output_data)
    print("System identification completed.")

    # Identify the ideal system using SRIM
    print("Identifying ideal system using SRIM method...")
    SRIM_ideal_system = simulation.identify_system_SRIM(input_data, ideal_output_data)
    print("Ideal system identification completed.")

    # Generate initial controller gain based on the order of the system
    initial_controller_gain = generate_initial_controller_gain(system_order)
    print(f"Initial controller gain generated: {initial_controller_gain}")

    # Load state feedback gain array from a predefined source or file
    state_feedback_gain_array = load_state_feedback_gain()
    print(f"State feedback gain array loaded: {state_feedback_gain_array}")

    # Plot the step response for the identified system
    print("Plotting step response for the identified system...")
    simulation.plot_step_response_PlantVsIdeal(SRIM_plant_system, SRIM_ideal_system)

    # Plot the eigenvalues for the identified system
    print("Plotting eigenvalues for the identified system...")
    simulation.plot_eigenvalues_PlantVsIdeal(SRIM_plant_system, SRIM_ideal_system)

    # Plot the Bode plot for the identified system
    print("Plotting Bode plot for the identified system...")
    simulation.plot_bode_PlantVsIdeal(SRIM_plant_system, SRIM_ideal_system)

    # Process the system matrix and save the natural frequencies
    print("Processing system matrix and saving natural frequencies...")
    simulation.process_matrix_and_save(SRIM_plant_system.A, filename="plant_system_eigenvalues_frequencies.csv")
    simulation.process_matrix_and_save(SRIM_ideal_system.A, filename="plant_system_eigenvalues_frequencies.csv")

    # Set up the State Feedback Controller
    print("Setting up State Feedback Controller...")
    SFC = StateFeedbackController(
        n=system_order, 
        plant_system=SRIM_plant_system, 
        ideal_system=SRIM_ideal_system, 
        input_signal=input_data, 
        test_signal=test_data, 
        sampling_rate=sampling_rate, 
        F_ini=initial_controller_gain, 
        F_ast = state_feedback_gain_array
    )

    # Simulate the system and get the output signals
    print("Running the State Feedback Controller simulation...")
    uncontrolled_output, controlled_output, control_input_signal = SFC.run()
    print("Simulation completed.")

    # Save the resulting audio signals
    print("Saving the simulated output signals as audio files...")
    al.save_audio(uncontrolled_output, sampling_rate, 'UncontrolledOutput')
    al.save_audio(controlled_output, sampling_rate, 'ControlledOutput')
    al.save_audio(control_input_signal, sampling_rate, 'ControlInputSignal')
    print("All audio files saved.")

if __name__ == "__main__":
    main()
