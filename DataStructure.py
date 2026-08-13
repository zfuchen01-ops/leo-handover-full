

class Stack(object):
  # 初始化栈为空列表
  def __init__(self):
    self.items = []
  
  def empty(self):
    return self.items == []
  
  def top(self):
    return self.items[len(self.items) - 1]
  
  def size(self):
    return len(self.items)
  
  def push(self, item):
    self.items.append(item)
  
  def pop(self):
    return self.items.pop()