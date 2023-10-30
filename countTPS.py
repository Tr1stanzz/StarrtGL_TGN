class countTPS:
    def __init__(self):
        self.wait_input = 0
        self.wait_fut = 0
        self.t_sample = 0
        self.t_union = 0
        self.t_get = 0
        self.count_union = 0
        self.count_sample = 0
        self.count_mess = 0
        self.t_mask = 0
        self.t_rpc = 0
    def print_(self):
        print('wait_input {} wait_fut {} t_sample {} t_union {} t_get {} count_union {} count_sample{} avg wait_input {} avg wait_fut {} avg t_sample {} avg t_union {} t_get {}'.
              format(self.wait_input,self.wait_fut,self.t_sample,self.t_union,self.t_get,self.count_union,self.count_sample,
                     self.wait_input/self.count_sample,self.wait_fut/self.count_union,self.t_sample/self.count_sample,self.t_union/self.count_union,self.t_get/self.count_sample))
    def print_t2(self):
        print('t_mask {} t_rpc {} count_mess {} avg mask {} avg rpc {}'.
              format(self.t_mask,self.t_rpc,self.count_mess,self.t_mask/self.count_mess,self.t_rpc/self.count_mess))

tps = countTPS()