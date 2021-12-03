import pandas as pd
import numpy as np

import pickle
import os

RAND_MEAN = 0.2
RAND_STD = 0.1

class Neuron():
  """
  Neuron
  """
  
  def __init__(self, num: int, name: str):
    """
    value: 0 to 1
    """
    self.num = num
    self.name = name
    self.value = 0.0 # initial value
    self.value_record = [] # updated value history [[time, value], ...]
    self.threshold = 1.0 # fire threshold
    self.targets = [] # [(target Neuron(), weight), ...] pair
    self.last_time_record = [0, 0] # last pair of [time, value] for efficient value storing
    self.current_layer = None # a layer this neuron belogns to
    
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
  
  def update_value(self, stim: float, t: float, weight_update: bool):
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
      self.fire(t, weight_update=weight_update)
      
      """
      reset current value
      """
      self.value = np.random.normal(RAND_MEAN, RAND_STD, size=(1))[0]
      pass
      
    return
    
    
  def fire(self, t: float, weight_update=True):
    """
    send pulses to other neurons
    """
    # print(f"{self.name} fire")
    weight_list = []
    for target_tuple in self.targets:
      target_neuron, weight = target_tuple
      target_neuron.update_value(weight, t, weight_update)
      weight_list.append(weight)
      pass
    
    if weight_update:
      self.update_weights(weight_list)
    
    return
  
  def update_weights(self, weight_list):
    """
    TODO: elaborate neighbouring neurons inhibitory algorithm
    
    inhibit neighbour neuron values
    collect neigbhour neurons by its number
    
    e.g. for neuron 37
    inhibit following neurons: 38, 39, 40, 41, 42
    set value to 0.0
    """
    
    
    """
    increase my target weights
    
    me:
    new_weight = old_weight + abs(0.5 * old_weight)
    """
    new_my_targets = []
    for target_tuple in self.targets:
      target_neuron, weight = target_tuple
      updated_weight = weight + abs(0.5 * weight)
      
      """ set weight boundary [-1.0, 1.0] """
      if updated_weight > 1.0:
        updated_weight = 1.0
      elif updated_weight < -1.0:
        updated_weight = -1.0
      new_my_targets.append((target_neuron, updated_weight))
    self.targets = new_my_targets
    
    
    
    """
    collect neighbouring neurons
    """
    neurons_in_the_same_layer = self.current_layer.neurons
    if self.num > len(neurons_in_the_same_layer) - 6:
      neighbour_neurons_tmp = list(neurons_in_the_same_layer[self.num + 1:self.num + (len(neurons_in_the_same_layer) - self.num)])
      neighbour_neurons = neighbour_neurons_tmp + list(neurons_in_the_same_layer[self.num - (5 - len(neighbour_neurons_tmp)):self.num - 1])
    else:
      neighbour_neurons = neurons_in_the_same_layer[self.num + 1:self.num + 6]
    
    
    """
    set neighbour neuron values zero
    update neighbour neuron weights 
    
    neighbours:
    new_weight = old_weight - abs(0.5 * old_weight)
    """
    for neighbour_neuron in neighbour_neurons:
      neighbour_neuron.value = 0.0
      each_targets = neighbour_neuron.targets
      new_targets = []
      for each_target in each_targets:
        _, weight = each_target
        weight = weight - abs(0.2*weight)
        """ set weight boundary [-1.0, 1.0] """
        if weight > 1.0:
          weight = 1.0
        elif weight < -1.0:
          weight = -1.0
        each_target = (_, weight)
        new_targets.append(each_target)
      neighbour_neuron.targets = new_targets
      
    """
    TODO: add update weights algorithm
    
    toy algorithm
    subtract min weigths from all the weights
    """
    # if len(weight_list) > 0:
    #   min_weight = min(weight_list)
    #   for target_tuple in self.targets:
    #     target_neuron, weight = target_tuple
    #     updated_weight = weight - min_weight
        
    #     """ set weight boundary [-1.0, 1.0] """
    #     if weight > 1.0:
    #       weight = 1.0
    #     elif weight < -1.0:
    #       weight = -1.0
          
    #     target_tuple = (target_neuron, updated_weight)
    #     pass
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
      neuron_obj = Neuron(i, f"{self.name} neuron {i}")
      neuron_obj.current_layer = self
      self.neurons.append(neuron_obj)
      
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
  def update_neurons(self, t=0, max_t=0, values=None, weight_update=True):
    """
    values: vector (m, 1), m: number of neurons
    [0.1, 0.2, 0, ...]
    """
    if t % 10 == 0:
      print(f"timestep: {t} {(100*t)/max_t}%")
      
    if len(self.neurons) != len(values):
      raise ValueError(f"the values length and the number of neurons has to be the same. values length: {len(values)}  number of neurons: {len(self.neurons)}")
    
    for neuron, val in zip(self.neurons, values):
      neuron.update_value(val, t, weight_update)
      
    return


  """
  """
  def save_activity_record_to_csv(self, end_t, dirname):
    
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
      
      
    pd.DataFrame(all_neuron_record).to_csv(f"/Users/takashimac/Documents/Python/Meta/export/network/{dirname}/{self.name}.csv")
      
    return
    










class Network():
  """
  a network of layers
  each layer contains hundreds of neurons
  """
  
  def __init__(self, name):
    self.name = name
    self.layers = [] # layers[0] is treated as the input layer and the last element is treated as the output layer
    self.data_shape = ()
    self.current_timesteps = 0
    
    try:
      os.makedirs(f"/Users/takashimac/Documents/Python/Meta/export/network/{self.name}")
    except FileExistsError:
      pass
    
    return


  def learn(self, data, weight_update=True):
    """
    insert data to the first layer
    
    data: (t, num of neurons in the layer1)
    weight_update: update weight or not during learning
    
    time 0 to T
    continually stimulate all the neurons
    """
    layer1 = self.layers[0]
    self.data_shape = np.array(data).shape
    for t, values in enumerate(data): # complexity O(n^2)
      layer1.update_neurons(t=t, max_t=data.shape[0], values=values, weight_update=weight_update)
      self.current_timesteps += 1
    
    return

  def save_output_layer(self, timesteps: int, MEMO: str):
    """
    save meta info
    save output layer activity from 0 to T
    """
    meta_info = {"name": self.name, "input_data_shape": str(self.data_shape), "current_timesteps": self.current_timesteps, "memo": MEMO}
    pd.DataFrame([meta_info]).to_csv(f"/Users/takashimac/Documents/Python/Meta/export/network/{self.name}/meta.csv")
    
    self.layers[len(self.layers) - 1].save_activity_record_to_csv(timesteps, self.name)
    return











class DataGenerator():
  
  
  def __init__(self):
    return
  
  def random(self, timepoints: int, num_of_neurons: int) -> np.array:
    """
    return random matrix (t, num of neurons)
    """
    print("loading random dataset")
    data = np.random.rand(timepoints, num_of_neurons)
    print(f"data shape {data.shape}")
    return data # matrix (t, num of neurons)

  def mnist(self, timepoints: int) -> np.array:
    """
    return matrix that contains mnist image data (num of images, image matrix)
    """
    print("loading mnist dataset")
    x_train = np.array(pd.read_csv('import/mnist.csv', index_col=0)) # train_X (60000, 28, 28)
    
    if timepoints <= 60000:
      data = x_train[0:timepoints, :]
    else:
      raise ValueError("timepoints should not exceed 60000")
      
    print(f"data shape {data.shape}")
    # img_data = x_train.reshape(60000, 28**2) # (60000, 784)
    return data
















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


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
  with open(filename, 'rb') as inp:
    obj = pickle.load(inp)
    return obj