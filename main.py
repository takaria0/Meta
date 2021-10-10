from src import Neuron, Layer, save_value_record_to_csv, save_input_data
import numpy as np
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
  """ 
  1. init Layer
  """
  layer1 = Layer("Input Layer")
  layer2 = Layer("Middle One Layer")
  layer3 = Layer("Middle Two Layer")
  layer4 = Layer("Output Layer")
  
  """
  2. initialize neurons
  """
  layer1.populate_neurons(784)
  layer2.populate_neurons(50)
  layer3.populate_neurons(200)
  layer4.populate_neurons(10)
  
  """
  3. connect neurons
  """
  layer1.set_target_neurons(layer2, target_selection="all")
  layer2.set_target_neurons(layer3, target_selection="sparse")
  layer3.set_target_neurons(layer4, target_selection="all")
  
  
  """
  4. prepare input data
  currently, 
  vector u (t, 1)
  
  change this to take diffrent values 
  vector u (t, m) 
  
  m: number of neurons in the layer
  t: time steps
  """
  timesteps = 100
  input_data_2 = np.random.rand(timesteps,784) # u (t, num_of_neurons)
  # (train_X, train_y), (test_X, test_y) = mnist.load_data(path='import/mnist.npz') # train_X (60000, 28, 28)
  # img_data = train_X.reshape(60000, 28**2) # (60000, 784)
  # input_data_3 = img_data[0:100, :]
  
  """
  5. fire, insert data to neurons
  time 0 to T
  continually stimulate all the neurons
  """
  for t, values in enumerate(input_data_2): # complexity O(n^2)
    layer1.update_neurons(t=t, values=values)
  
  
  layer4.save_activity_record_to_csv(timesteps)
  
  return



def main():
  # test_run_each_neuron_v1()
  test_run_layer_system_v2()
  
  return
  
  
main()


