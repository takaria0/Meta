from src import Neuron, Layer, save_value_record_to_csv, save_input_data

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
  init Layer
  """
  layer1 = Layer("Input Layer")
  layer2 = Layer("Middle Layer")
  layer3 = Layer("Output Layer")
  
  layer1.populate_neurons(1000)
  layer2.populate_neurons(10)
  layer3.populate_neurons(20)
  
  layer1.set_target_neurons(layer2)
  layer2.set_target_neurons(layer3)
  
  
  """
  prepare input data
  """
  a = 0.4
  input_data = [0,0,a,a,0,0,a,0,a,a]*10
  
  """
  fire
  """
  layer1.update_neurons(values=input_data)
  
  layer3.save_activity_record()
  
  return



def main():
  # test_run_each_neuron_v1()
  test_run_layer_system_v2()
  
  return
  
  
main()


