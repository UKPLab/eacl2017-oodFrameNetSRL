class Config:  # Container class for configurations
    def __init__(self, clf, feature_extractor, lexicon, vsm, multiword_averaging,
                 all_unknown, num_components, max_sampled, num_epochs):
        self.clf = clf
        self.feat_extractor = feature_extractor
        self.lexicon = lexicon
        self.vsm = vsm
        self.multiword_averaging = multiword_averaging
        self.all_unknown = all_unknown
        self.num_components = num_components
        self.max_sampled = max_sampled
        self.num_epochs = num_epochs

    def get_clf(self):
        return self.clf

    def get_feat_extractor(self):
        return self.feat_extractor

    def get_lexicon(self):
        return self.lexicon

    def get_vsm(self):
        return self.vsm

    def get_multiword_averaging(self):
        return self.multiword_averaging

    def get_all_unknown(self):
        return self.all_unknown
    
    def get_num_components(self):
        return self.num_components
    
    def get_max_sampled(self):
        return self.max_sampled
    
    def get_num_epochs(self):
        return self.num_epochs

    def __str__(self):
        return "c_"+self.clf.__name__+"__"+"f_"+self.feat_extractor.__name__+"__"+\
               "l_"+(self.lexicon if self.lexicon is not None else "NA") +"__"+"vsm_"+\
               (self.vsm if self.vsm is not None else "NA")+\
              "__"+"MWA_"+str(self.multiword_averaging)+"__unk_"+str(self.all_unknown)+\
              "__comp_"+str(self.num_components)+"__samp_"+str(self.max_sampled)+"__ep_"+str(self.num_epochs)
