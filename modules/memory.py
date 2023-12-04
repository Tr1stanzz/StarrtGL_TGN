import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

  def __init__(self, memory_dimension, input_dimension, message_dimension=None,
               device="cpu", combination_method='sum'):
    super(Memory, self).__init__()
    self.memory_dimension = memory_dimension
    self.input_dimension = input_dimension
    self.message_dimension = message_dimension
    self.device = device

    self.combination_method = combination_method
  
  def load_memory_from_mail(self, mailbox=None, node_ids=None):
    if mailbox is not None:
       with torch.no_grad():
        memory,memory_ts,mail,mail_ts = mailbox.get_memory_by_scatter(node_ids)
        # print(torch.equal(memory_ts,mail_ts))
        self.memory = memory.clone()
        self.last_update = memory_ts.clone()
        self.mail = mail.clone()
        self.mail_ts = mail_ts.clone()

  def store_raw_messages(self, nodes, node_id_to_messages):
    for node in nodes:
      self.messages[node].extend(node_id_to_messages[node])

  def get_memory(self, node_idxs):
    return self.memory[node_idxs, :]

  def set_memory(self, node_idxs, values):
    self.memory = self.memory.clone()
    self.memory[node_idxs, :] = values

  def get_last_update(self, node_idxs):
    return self.last_update[node_idxs]

  def backup_memory(self):
    messages_clone = {}
    for k, v in self.messages.items():
      messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

    return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

  def restore_memory(self, memory_backup):
    self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

    self.messages = defaultdict(list)
    for k, v in memory_backup[2].items():
      self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

  def detach_memory(self):
    self.memory.detach_()

    # Detach all stored messages
    for k, v in self.messages.items():
      new_node_messages = []
      for message in v:
        new_node_messages.append((message[0].detach(), message[1]))

      self.messages[k] = new_node_messages

  def clear_messages(self, nodes):
    for node in nodes:
      self.messages[node] = []
