import logging
import os
import time

class Config():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 添加控制台管理器(即控制台展示log内容）
    ls = logging.StreamHandler()
    ls.setLevel(logging.DEBUG)

    # 设置log的记录格式
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(filename)s-[line:%(lineno)d]''-%(levelname)s: %(message)s')

    # 把格式添加到控制台管理器,即控制台打印日志
    ls.setFormatter(formatter)
    # 把控制台添加到logger
    logger.addHandler(ls)

    # 先在项目目录下建一个logs目录，来存放log文件（可自定义路径）
    print(os.path.abspath('.') )
    logdir = os.path.join(os.path.abspath('.'), 'logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    # 再在logs目录下创建以日期开头的.log文件
    logfile = os.path.join(logdir, time.strftime('%Y-%m-%d %H:%M:%S') + '.log')

    # 添加log的文件处理器，并设置log的配置文件模式编码
    lf = logging.FileHandler(filename=logfile, encoding='utf8')
    # 设置log文件处理器记录的日志级别
    lf.setLevel(logging.DEBUG)
    # 设置日志记录的格式
    lf.setFormatter(formatter)
    # 把文件处理器添加到log
    logger.addHandler(lf)

    def get_config(self):
        return self.logger


logger = Config().get_config()
