import pandas as pd
import numpy as np

class Neuron():
  """
  Neuron
  """
  
  def __init__(self, name: str):
    """
    value: 0 to 1
    """
    self.name = name
    self.value = 0.0 # initial value
    self.value_record = [] # value history
    self.threshold = 1.0 # fire threshold
    self.targets = [] # (target Neuron class, weight) pair
    
    return
    
    
  def init_targets(self, targets: list):
    """
    initialize target neurons
    """
    assert type(targets) == list
    
    self.targets = targets
    return
  
  def update_value(self, stim: float, t: float):
    """
    accumulate value for the neuron
    
    stim: float
    """
    self.value += stim
    self.value_record.append([t, self.value])
    
    """fire"""
    if self.value >= self.threshold:
      self.fire(t)
      
      """reset value"""
      rand_mean = 0.2
      rand_std = 0.1 
      self.value = np.random.normal(rand_mean, rand_std, size=(1))[0]
      pass
      
    return
    
    
  def fire(self, t: float):
    """
    send pulses to other neurons
    """
    print(f"{self.name} fire")
    for target_tuple in self.targets:
      target_neuron, weight = target_tuple
      target_neuron.update_value(weight, t)
      pass
    
    return
  

class Layer():
  """
  layer of multiple neurons
  contain multiple Neuron classes
  """
  def __init__(self, name=""):
    self.neurons = []
    self.name = name
    self.num_of_neurons = 0
    return
  
  """
  1. populate neurons in the layer
  """
  def populate_neurons(self, num_of_neurons):
    self.num_of_neurons = num_of_neurons
    for i in range(self.num_of_neurons):
      self.neurons.append(Neuron(f"{self.name} neuron {i}"))
    return
  
  """
  2. set all the neurons in the next layer as target neurons for each neuron in the current layer
  """
  def set_target_neurons(self, target_layer):
    target_neurons = target_layer.neurons
    
    """initialize weights for all the target neurons """
    target_neurons_and_weights_pair = [] # this is the actual target list, contains a tuple of (neuron, weight)
    for target_neuron in target_neurons:
      weight = 0.1 # replace some random variables later on
      target_neuron_weight_pair = (target_neuron, weight)
      target_neurons_and_weights_pair.append(target_neuron_weight_pair)
    
    """set target neurons"""
    for neuron in self.neurons:
      neuron.init_targets(target_neurons_and_weights_pair)
    return
  
  """
  only for Input Layer
  input data and start firing all the neurons!
  """
  def update_neurons(self, values=None):
    """
    e.g. same stimuli 0.4 for 1000 timesteps
    value = [0.4]*1000
    """    
    # if len(values) != len(self.neurons):
    #   raise ValueError("")
    
    """
    time 0 to T
    continually stimulate all the neurons
    """
    for t, value in enumerate(values):
      for neuron in self.neurons:
        neuron.update_value(value, t)
    return


  """
  """
  def save_activity_record(self):
    
    for neuron in self.neurons:
      each_neuron_record = neuron.value_record
      pd.DataFrame(each_neuron_record).to_csv(f"/Users/takashimac/Documents/Python/Meta/export/layer/{neuron.name}.csv")
      
    return
    



      
class MicroCircuit():
  """
  create graph structure for initializing target neurons
  """

  def __init__(self):
    value = 0
    to = []
    
    
class MicroGraph():
  
  pass


def save_input_data(data):
  """
  """
  data = pd.DataFrame(data)
  data.to_csv("/Users/takashimac/Documents/Python/Meta/export/input.csv")
  return

def save_value_record_to_csv(value_record):
  """
  """
  
  data = pd.DataFrame(value_record)
  data.to_csv("/Users/takashimac/Documents/Python/Meta/export/neural_activity.csv")
  return