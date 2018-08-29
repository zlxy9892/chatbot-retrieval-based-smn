#encoding:utf-8

import jieba
import codecs
import re

class Seg(object):
    
    def __init__(self, file_stopwords="./data/stopword.txt", file_userdict="./data/dict_for_cutword.txt"):
        self.stopwords = []
        self.stopword_filepath = file_stopwords
        self.userdict_filepath = file_userdict
        self.read_in_stopword()
        self.read_in_userdict()

    def read_in_stopword(self): # 读入停用词
        if self.stopword_filepath is not None:
            file_obj = codecs.open(self.stopword_filepath, 'r', 'utf-8')
            while True:
                line = file_obj.readline()
                line = line.strip('\r\n')
                if not line:
                    break
                self.stopwords.append(line)
            file_obj.close()

    def read_in_userdict(self):    # 读入自定义词典
        if self.userdict_filepath is not None:
            jieba.load_userdict(self.userdict_filepath)
    
    def replace_special_words0(self, content):
        content = re.sub(re.compile(r'<span.*/span>'), 'WASTE1', content)
        content = re.sub(re.compile(r'<spandata-hidden="\[数字x\]"data-rel="sendLinkOid"class="J_BarRelCase">'), 'WASTE1', content)
        content = re.sub(re.compile('<s>'), '|', content)
        content = re.sub(re.compile(r'https?://item.jd.com/[0-9]*.html|https?://item.jd.com/\[数字x\].html'), 'LINK1', content)
        content = re.sub(re.compile(r'https?://vc.jd.com/sampling.html'), 'LINK2', content)
        content = re.sub(re.compile(r'https?://m-eve.jd.com/dxtyk/index体验卡'), 'LINK3', content)
        content = re.sub(re.compile(r'https?://\[链接x\]'), 'LINK4', content)
        content = re.sub(re.compile(r'有什么问题我可以帮您处理或解决呢?'), 'COMMON1', content)
        content = re.sub(re.compile(r'请问还有其他还可以帮到您的吗?'), 'COMMON2', content)
        content = re.sub(re.compile(r'亲爱的，还有其他业务信息需要妹子为您查询的吗?'), 'COMMON2', content)
        content = re.sub(re.compile(r'咨询订单号:\[数字x\]订单金额:\[金额x\]下单时间:\[日期x\]|咨询订单号:\[ORDERID_[0-9]*\]订单金额:\[金额x\]下单时间:\[日期x\]'), 'SPWORD1', content)
        content = re.sub(re.compile(r'\[订单编号:\[数字x\]，订单金额:\[金额x\]，下单时间:\[日期x\]\[时间x\]\]'), 'SPWORD2', content)
        content = re.sub(re.compile(r'#E-s\[数字x\]|#e-s\[数字x\]'), 'SPWORD3', content)
        content = re.sub(re.compile(r'\[数字x\]'), 'SPWORD4', content)
        content = re.sub(re.compile(r'¥?\[金额x\]'), 'SPWORD5', content)
        content = re.sub(re.compile(r'\[日期x\]'), 'SPWORD6', content)
        content = re.sub(re.compile(r'\[时间x\]'), 'SPWORD7', content)
        content = re.sub(re.compile(r'\[姓名x\]'), 'SPWORD8', content)
        content = re.sub(re.compile(r'\[站点x\]'), 'SPWORD9', content)
        content = re.sub(re.compile(r'\[地址x\]'), 'SPWORD10', content)
        content = re.sub(re.compile(r'\[电话x\]'), 'SPWORD11', content)
        content = re.sub(re.compile(r'\[邮箱x\]'), 'SPWORD12', content)
        content = re.sub(re.compile(r'\[组织机构x\]'), 'SPWORD13', content)
        content = re.sub(re.compile(r'\[ORDERID_[0-9]*\]'), 'ORDER1', content)
        return content

    def replace_special_words(self, content):
        content = re.sub(re.compile(r'<span.*/span>'), 'WASTE1', content)
        content = re.sub(re.compile(r'<spandata-hidden="\[数字x\]"data-rel="sendLinkOid"class="J_BarRelCase">'), 'WASTE1', content)
        # content = re.sub(re.compile(r'人工客服哦#E-s\[数字x\]\)#E-s\[数字x\]#E-s\[数字x\]#E-s\[数字x\]'), 'WASTE1', content)
        content = re.sub(re.compile('<s>'), '|', content)

        content = re.sub(re.compile(r'(请问)?有什么(能)?(可以)?帮到您'), 'COMMON1', content)
        content = re.sub(re.compile(r'有什么问题我可以帮您处理或解决呢'), 'COMMON1', content)
        content = re.sub(re.compile(r'请问(亲爱的)?有什么可以为您效劳的呢|(亲爱的.?)请问还有其他可以帮到您'), 'COMMON1', content)
        content = re.sub(re.compile(r'有什么问题小李子可以帮您处理或解决呢'), 'COMMON1', content)
        content = re.sub(re.compile(r'请问还有其他还可以帮到您的吗'), 'COMMON2', content)
        content = re.sub(re.compile(r'亲爱的，还有其他业务信息需要妹子为您查询的吗'), 'COMMON2', content)
        content = re.sub(re.compile(r'(尊敬的商家您好，?)?我是您的京东物流小红人工号\[数字x\]'), 'COMMON3', content)
        content = re.sub(re.compile(r'(小妹(这边)?正在)?(请稍等一下哦.?)?火速为您查询|(这边)?(小妹)?(就)?帮您查询|(请稍等一下哦.?)?小妹正在查询中|(请稍等一下哦.?)?小妹正在飞奔为您查询|((稍等)?小妹)?帮您看一下哈'), 'COMMON4', content)
        content = re.sub(re.compile(r'(亲爱的客户.?)?辛苦您稍等一下下|(好的.?)稍等'), 'COMMON4', content)
        content = re.sub(re.compile(r'可在手机端打开https?://\[链接x\]或在电脑端打开https?://myivc.jd.com/fpzz.html进行发票查询和下载'), 'COMMON5', content)
        content = re.sub(re.compile(r'请问您是要咨询订单:\[ORDERID_[0-9]*\]'), 'COMMON6', content)
        content = re.sub(re.compile(r'正在为您核实处理'), 'COMMON7', content)
        content = re.sub(re.compile(r'申请路径:\[站点x\]可通过“我的京东”-“客户服务”-“返修退换货”内申请\(也可直接点击此链接:https?://myjd.jd.com/repair/orderlist.action;【APP端】可通过“我的”-“客户服务”-“退换/售后“中申请~(;\[站点x\]可通过个人中心-客户服务-退换/售后进行申请哦~)?'), 'COMMON8', content)

        content = re.sub(re.compile(r'https?://item.jd.com/[0-9]*.html|https?://item.jd.com/\[数字x\].html'), 'LINK1', content)
        content = re.sub(re.compile(r'https?://vc.jd.com/sampling.html'), 'LINK2', content)
        content = re.sub(re.compile(r'https?://m-eve.jd.com/dxtyk/index体验卡'), 'LINK3', content)
        content = re.sub(re.compile(r'https?://\[链接x\]'), 'LINK4', content)
        content = re.sub(re.compile(r'https?://myjd.jd.com/repair/orderlist.action'), 'LINK5', content)
        content = re.sub(re.compile(r'https?://myivc.jd.com/fpzz.html'), 'LINK6', content)
        content = re.sub(re.compile(r'https?://rec.ql.jd.com/price/soplbpprice'), 'LINK7', content)

        content = re.sub(re.compile(r'咨询订单号:\[数字x\]订单金额:\[金额x\]下单时间:\[日期x\]|咨询订单号:\[ORDERID_[0-9]*\]订单金额:\[金额x\]下单时间:\[日期x\]|咨询订单号:\[ORDERID_[0-9]*\]商品ID:[0-9]*'), 'SPWORD1', content)
        content = re.sub(re.compile(r'\[订单编号:\[ORDERID_[0-9]*\]，订单金额:\[金额x\]，下单时间:\[日期x\]\[时间x\]\]'), 'SPWORD1', content)
        content = re.sub(re.compile(r'\[订单编号:\[数字x\]，订单金额:\[金额x\]，下单时间:\[日期x\]\[时间x\]\]'), 'SPWORD1', content)
        content = re.sub(re.compile(r'顾客通过点击web咚咚\[站点x\]信息发送:\[订单编号:\[ORDERID_[0-9]*\]，订单金额:\[金额x\]，下单时间:\[日期x\]\[时间x\]\]'), 'SPWORD2', content)
        content = re.sub(re.compile(r'顾客通过点击web咚咚\[站点x\]信息发送:'), 'SPWORD2', content)
        content = re.sub(re.compile(r'#E-s\[数字x\]|#e-s\[数字x\]'), 'SPWORD4', content)
        content = re.sub(re.compile(r'\[数字x\]'), 'SPWORD4', content)
        content = re.sub(re.compile(r'¥?\[金额x\]'), 'SPWORD5', content)
        content = re.sub(re.compile(r'\[日期x\]'), 'SPWORD6', content)
        content = re.sub(re.compile(r'\[时间x\]'), 'SPWORD7', content)
        content = re.sub(re.compile(r'\[姓名x\]'), 'SPWORD8', content)
        content = re.sub(re.compile(r'\[站点x\]'), 'SPWORD9', content)
        content = re.sub(re.compile(r'\[地址x\]'), 'SPWORD10', content)
        content = re.sub(re.compile(r'\[电话x\]'), 'SPWORD11', content)
        content = re.sub(re.compile(r'\[邮箱x\]'), 'SPWORD12', content)
        content = re.sub(re.compile(r'\[组织机构x\]'), 'SPWORD13', content)

        content = re.sub(re.compile(r'\[ORDERID_[0-9]*\]'), 'ORDER1', content)
        return content

    def cut(self, sentence, stopword=True):
        sentence = self.replace_special_words(sentence)
        seg_list = jieba.cut(sentence)  # 切词
        results = []
        if stopword == False:
            results = list(seg_list)
        else:
            for seg in seg_list:
                if seg in self.stopwords and stopword:
                    continue    # 去除停用词
                results.append(seg)
        return results

    def cut_for_search(self, sentence, stopword=True):
        sentence = self.replace_special_words(sentence)
        seg_list = jieba.cut_for_search(sentence)
        results = []
        if stopword == False:
            results = list(seg_list)
        else:
            for seg in seg_list:
                if seg in self.stopwords and stopword:
                    continue    # 去除停用词
                results.append(seg)
        return results
