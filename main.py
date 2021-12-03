from src import Neuron, Layer, save_value_record_to_csv, save_input_data, DataGenerator, save_object, Network
import numpy as np

from datetime import datetime
# from keras.datasets import mnist

def test_run_each_neuron_v1():
  """
  initialize neuron
  
  input: 1, 2, 3 # Layer 1
  middle: 4, # Layer 2
  output: 5, 6 # Layer 3
  """
  neuron_1 = Neuron("1")
  neuron_2 = Neuron("2")
  neuron_3 = Neuron("3")
  
  neuron_4 = Neuron("4")
  
  neuron_5 = Neuron("5")
  neuron_6 = Neuron("6")
  
  
  """
  create a network
  """
  neuron_1.init_targets([neuron_4])
  neuron_2.init_targets([neuron_4])
  neuron_3.init_targets([neuron_4])
  
  neuron_4.init_targets([neuron_5, neuron_6])
  
  """
  prepare input data
  """
  a = 0.4
  input_data = [0,0,a,a,0,0,a,0]*5
  
  
  """
  run simulation
  """
  value_record = []
  for data in input_data:
    
    neuron_1.update_value(data)
    neuron_2.update_value(data)
    neuron_3.update_value(data)
    
    value_record.append([neuron_1.value, neuron_2.value, neuron_3.value, neuron_4.value, neuron_5.value, neuron_6.value,])
    pass
  
  
  """
  save data
  """
  save_input_data(input_data)
  save_value_record_to_csv(value_record)
  return

def test_run_layer_system_v2():
  MEMO = "inhibit_neighbour_neurons"
  NUM_OF_INPUT_NEURONS = 784
  NUM_OF_OUTPUT_NEURONS = 10
  TIMESTEPS = 50000
  """ 
  1. init Layer
  """
  layer1 = Layer("Input Layer")
  layer2 = Layer("Hidden One Layer")
  layer3 = Layer("Output Layer")
  
  network_name = f"{datetime.today().strftime('%Y%m%d_%H%M%S')}_{TIMESTEPS}_{NUM_OF_INPUT_NEURONS}_{MEMO}_mnist_network"
  mnist_network = Network(network_name)
  mnist_network.layers = [layer1, layer2, layer3]
  
  """
  2. initialize neurons
  """
  layer1.populate_neurons(NUM_OF_INPUT_NEURONS)
  layer2.populate_neurons(100)
  layer3.populate_neurons(NUM_OF_OUTPUT_NEURONS)
  
  
  
  """
  3. connect neurons
  """
  # Forward
  layer1.set_target_neurons(layer2, target_selection="all")
  layer2.set_target_neurons(layer3, target_selection="all")
  
  
  # Backward (need computing power, nearly impossible, find another weights update rule) 
  # maybe not always? connect backward during some time.
  # layer5 = Layer("Backward Layer")
  # layer5.populate_neurons(78)
  # import sys; sys.setrecursionlimit(10**5)
  # layer3.set_target_neurons(layer1, target_selection="sparse") # 
  # layer4.set_target_neurons(layer5, target_selection="sparse") #
  # layer5.set_target_neurons(layer1, target_selection="sparse")
  
  
  """
  4. prepare input data
  currently, 
  vector u (t, 1)
  
  change this to take diffrent values 
  vector u (t, m) 
  
  m: number of neurons in the layer
  t: time steps
  """
  data = DataGenerator()
  input_data = data.mnist(TIMESTEPS) # u (t, num_of_neurons)
  
  """
  5. fire, insert data to neurons
  time 0 to T
  continually stimulate all the neurons
  """
  mnist_network.learn(input_data)
  
  """
  6. save output neurons activity data
  """
  mnist_network.save_output_layer(TIMESTEPS, MEMO)
  
  """
  7. save entire network
  """
  save_object(mnist_network, f"/Users/takashimac/Documents/Python/Meta/export/network/{mnist_network.name}/mnist_network_v1.pkl")
  return



def main():
  # test_run_each_neuron_v1()
  test_run_layer_system_v2()
  
  return
  
  
main()


