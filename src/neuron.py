import pandas as pd
import numpy as np

RAND_MEAN = 0.2
RAND_STD = 0.1

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
    self.value_record = [] # updated value history [[time, value], ...]
    self.threshold = 1.0 # fire threshold
    self.targets = [] # [(target Neuron(), weight), ...] pair
    self.last_time_record = [0, 0] # last pair of [time, value] for efficient value storing
    
    self.rand_mean = 0.2
    self.rand_std = 0.1 
    self.random = np.random.normal
    return
    
    
  def init_targets(self, targets: list):
    """
    initialize target neurons
    targets: [neuron: class, weight: float]
    """
    assert type(targets) == list
    
    self.targets = targets
    return
  
  def update_value(self, stim: float, t: float):
    """
    accumulate value for the neuron
    save the value
    update the time
    stim: float
    """
    
    """
    update current value of the neuron
    """
    self.value += stim
    
    """
    update value history, save the last record for efficiency
    """
    last_time = self.last_time_record[0]
    last_value = self.last_time_record[1]
    if last_time < t:
      self.value_record.append([last_time, last_value])
      self.last_time_record = [t, self.value]
    
    """
    fire (update target neurons)
    """
    if self.value >= self.threshold:
      self.fire(t)
      
      """
      reset current value
      """
      self.value = np.random.normal(RAND_MEAN, RAND_STD, size=(1))[0]
      pass
      
    return
    
    
  def fire(self, t: float):
    """
    send pulses to other neurons
    """
    # print(f"{self.name} fire")
    for target_tuple in self.targets:
      target_neuron, weight = target_tuple
      target_neuron.update_value(weight, t)
      pass
    
    """
    TODO: add update weights algorithm
    """
    
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
  
  TODO: instead of connecting all the neurons, add options to choose target neurons
  """
  def set_target_neurons(self, target_layer, target_selection="all"):
    
    target_neurons = target_layer.neurons
    
    """
    set target neurons
    currently, connect all the neurons (dense layer)
    """
    if target_selection == "all":
      """
      initialize weights for all the target neurons
      """
      target_neurons_and_weights_pair = [] # this is the actual target list, contains a tuple of (neuron, weight)
      for target_neuron in target_neurons:
        weight = np.random.normal(RAND_MEAN, RAND_STD, size=(1))[0] # replace some random variables later on
        target_neuron_weight_pair = (target_neuron, weight)
        target_neurons_and_weights_pair.append(target_neuron_weight_pair)
      
      """
      set target neurons to each neuron in the layer
      """
      for neuron in self.neurons:
        neuron.init_targets(target_neurons_and_weights_pair)
        
    elif target_selection == "sparse":
      """
      initialize weights for half the target neurons
      """
      target_neurons_and_weights_pair = [] # this is the actual target list, contains a tuple of (neuron, weight)
      for i, target_neuron in enumerate(target_neurons):
        if i % 2 == 0:
          weight = np.random.normal(RAND_MEAN, RAND_STD, size=(1))[0] # replace some random variables later on
          target_neuron_weight_pair = (target_neuron, weight)
          target_neurons_and_weights_pair.append(target_neuron_weight_pair)
      
      for neuron in self.neurons:
        neuron.init_targets(target_neurons_and_weights_pair)
      pass
    
    
    
    else:
      for neuron in self.neurons:
        neuron.init_targets(target_neurons_and_weights_pair)
      
    return
  
  """
  only for Input Layer
  input data and start firing all the neurons!
  """
  def update_neurons(self, t=0, values=None):
    """
    values: vector (m, 1), m: number of neurons
    [0.1, 0.2, 0, ...]
    """
    if t % 10 == 0:
      print(f"timestep: {t}")
      
    if len(self.neurons) != len(values):
      raise ValueError("the values length and the number of neurons has to be the same")
    
    for neuron, val in zip(self.neurons, values):
      neuron.update_value(val, t)
    return


  """
  """
  def save_activity_record_to_csv(self, end_t):
    
    all_neuron_record = []
    for neuron in self.neurons:
      each_neuron_record = neuron.value_record
      
      """
      insert missing timestep and value (as zero)
      """
      all_timesteps = {data[0]: data[1] for data in each_neuron_record} # {timesteps: value]
      interpolated_record = []
      for i in range(end_t):
        if i not in all_timesteps.keys():
          interpolated_record.append(0)
        else:
          interpolated_record.append(all_timesteps[i])
      
      all_neuron_record.append(interpolated_record)
      
      
    pd.DataFrame(all_neuron_record).to_csv(f"/Users/takashimac/Documents/Python/Meta/export/layer/{self.name}.csv")
      
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