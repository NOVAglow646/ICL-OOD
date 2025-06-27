cd ./src/exp_for_theory
python BayesianSimulation_Preprocess.py --mode low_test_error_preference
python BayesianSimulation_Visualize.py --mode low_test_error_preference


cd ./src/exp_for_theory
python BayesianSimulation_Preprocess.py --mode similar_input_distribution_preference
python BayesianSimulation_Visualize.py --mode similar_input_distribution_preference