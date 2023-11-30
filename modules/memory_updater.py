from torch import nn
import torch


class MemoryUpdater(nn.Module):
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    pass


class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(SequenceMemoryUpdater, self).__init__()
    self.memory = memory
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device

  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    memory = self.memory.get_memory(unique_node_ids)
    self.memory.last_update[unique_node_ids] = timestamps

    updated_memory = self.memory_updater(unique_messages, memory)

    self.memory.set_memory(unique_node_ids, updated_memory)

  # def update_memory_by_mailbox(self, unique_node_ids):
  #   if len(unique_node_ids) <= 0:
  #     return

  #   memory = self.memory.get_memory(unique_node_ids)
  #   self.memory.last_update[unique_node_ids] = self.memory.mail_ts[unique_node_ids]
  #   updated_memory = self.memory_updater(self.memory.mail[unique_node_ids].squeeze(1), memory)
  #   self.memory.set_memory(unique_node_ids, updated_memory)

  def get_updated_memory_by_mailbox(self, unique_node_ids, time_encoder):
    if len(unique_node_ids) <= 0:
      # print("WTF?!")
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
    
    delta_time = self.memory.mail_ts[unique_node_ids] - self.memory.last_update[unique_node_ids]
    time_encoding = time_encoder(delta_time.unsqueeze(1)).view(len(unique_node_ids), -1)
    mails = self.memory.mail[unique_node_ids].squeeze(1)
    mails = torch.cat([mails, time_encoding], dim=1)
    self.memory.memory[unique_node_ids] = self.memory_updater(mails, self.memory.memory[unique_node_ids])

    self.memory.last_update[unique_node_ids] = self.memory.mail_ts[unique_node_ids]
    return self.memory.memory, self.memory.last_update

  def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    updated_memory = self.memory.memory.data.clone()
    updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

    updated_last_update = self.memory.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps

    return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
  if module_type == "gru":
    return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "rnn":
    return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
