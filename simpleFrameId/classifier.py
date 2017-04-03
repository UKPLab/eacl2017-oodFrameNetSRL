import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils.np_utils import to_categorical
from collections import Counter

from lightfm import LightFM
# LightFM source code had to be hacked as it is buggy and does not say with which python version it actually should work
        # aMatrix.tocsr() --> sp.csr_matrix(aMatrix)
        # aMatrix.tocoo() --> sp.coo_matrix(aMatrix)
from sklearn.metrics.pairwise import cosine_similarity


# Generic classifier, doesn't do much
class Classifier:
    def __init__(self, lexicon, all_unknown=False, num_components=False, max_sampled=False, num_epochs=False ):
        self.clf = None
        self.lexicon = lexicon
        self.all_unknown = all_unknown
        self.num_components = num_components
        self.max_sampled = max_sampled
        self.num_epochs = num_epochs

    def train(self, X, y, lemmapos):
        raise NotImplementedError("Not implemented, use child classes")
    def predict(self, X, lemmapos):
        raise NotImplementedError("Not implemented, use child classes")


# Data-driven majority baseline
class DataMajorityBaseline(Classifier):
    def train(self, X, y, lemmapos_list):
        self.majorityClasses = {}
        total_y = []
        # get frame by LU counts from DATA. Not seen in data = doesn't exist
        for X_i, y_i, lemmapos_i in zip(X, y, lemmapos_list):
            self.majorityClasses[lemmapos_i] = self.majorityClasses.get(lemmapos_i, []) + [y_i]
            total_y += [y_i]

        uninformed_majority = Counter(total_y).most_common(1)[0][0]  # uninformed majority for lemmas not seen in data

        # get top frame for each LU
        for lemmapos in self.majorityClasses:
            if len(self.majorityClasses.get(lemmapos, [])) == 0:
                self.majorityClasses[lemmapos] = uninformed_majority
            else:
                self.majorityClasses[lemmapos] = Counter(self.majorityClasses[lemmapos]).most_common(1)[0][0]

        self.majorityClasses["__UNKNOWN__"] = uninformed_majority
        print self.majorityClasses
        print "Majority baseline extracted, uninformed majority class is", uninformed_majority, ":", self.lexicon.idToFrame[uninformed_majority]

    def predict(self, X, lemmapos):
        if self.all_unknown:
            return self.majorityClasses["__UNKNOWN__"]
        return self.majorityClasses.get(lemmapos, self.majorityClasses["__UNKNOWN__"])


# Lexicon-driven majority baseline
class LexiconMajorityBaseline(DataMajorityBaseline):
    def train(self, X, y, lemmapos_list):
        frame_counts = []
        for y_i, lemmapos_i in zip(y, lemmapos_list):  # collect TOTAL frame counts from data
            frame_counts += [y_i]

        frame_counts = Counter(frame_counts)

        self.majorityClasses = {}
        uninformed_majority = frame_counts.most_common(1)[0][0]
        self.majorityClasses["__UNKNOWN__"] = uninformed_majority

        for lemmapos in self.lexicon.frameLexicon:   # for each lemma in LEXICON, determine most frequent frame among available, based on data
            available_frames = self.lexicon.get_available_frame_ids(lemmapos)
            available_frame_counts = Counter({f:frame_counts.get(f, 0) for f in available_frames})  # no frame in data - count set to 0
            self.majorityClasses[lemmapos] = available_frame_counts.most_common(1)[0][0]

        print "Majority baseline extracted, uninformed majority class is", uninformed_majority, ":", self.lexicon.idToFrame[uninformed_majority]


# A simple NN-based classifier
class SharingDNNClassifier(Classifier):
    def train(self, X, y, lemmapos_list):
        self.clf = Sequential()
        self.clf.add(Dense(256, input_dim=len(X[0]), activation='relu'))
        self.clf.add(Dense(100, activation='relu'))
        self.clf.add(Dense(output_dim=np.max(y)+1, activation='softmax'))  # np.max()+1 because frames are 0-indexed

        self.clf.compile(optimizer='adagrad',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        self.clf.fit(X, to_categorical(y, np.max(y)+1), verbose=1, nb_epoch=100)

    def predict(self, X, lemmapos):
        available_frames = self.lexicon.get_available_frame_ids(lemmapos)  # get available frames from lexicon
        ambig = self.lexicon.is_ambiguous(lemmapos)
        unknown = self.lexicon.is_unknown(lemmapos)  # unknown = not in lexicon

        if unknown or self.all_unknown:  # the all_unknown setting renders all lemma.pos unknown!
            available_frames = self.lexicon.get_all_frame_ids()  # if the lemma.pos is unknown, search in all frames
        else:
            # if the LU is known and has only one frame, just return it. Even if there is no data for this LU (!)
            if not ambig:
                return available_frames[0]

        y = self.clf.predict(X.reshape((-1, len(X))))[0]
        # pick the best-scoring frame among available
        bestScore = None
        bestClass = None
        for cl in available_frames:
            score = y[cl]
            if bestScore is None or score >= bestScore:
                bestScore = score
                bestClass = cl
        return bestClass


# classification with WSABIE latent representations
class WsabieClassifier(Classifier):
    def train(self, X, y, lemmapos_list):
        
        # MODEL
        self.clf = LightFM(no_components = self.num_components, learning_schedule = 'adagrad', loss = 'warp', \
                           learning_rate = 0.05, epsilon = 1e-06, item_alpha = 0.0, user_alpha = 1e-6, \
                           max_sampled = self.max_sampled, random_state = None)
        
        # DATA
        # training data
        # X: list of vectors
        #    each vector is the initial representation for a sentence (more precisely, for a predicate with context)
        #    --> these are the user features in the training set
        # y: list of IDs for frames
        #    the frame IDs are the labels for the representations
        #    --> these are used to create the interaction matrix for the training set such that LightFM can deal with it
        # y_interactionLabels: interaction matrix is of size (num sentences in y) x (num frames) with 1 indicating the frame label for a predicate in its context sentence
        y_interactionLabels = self.createInteractionMatrix(y)
                 
        # FIT
        self.clf = self.clf.fit(interactions = y_interactionLabels, user_features = X, item_features = None, \
                                sample_weight = None, epochs = self.num_epochs, num_threads = 2, verbose = True)

    def predict(self, X, lemmapos):
        # DATA
        # test data
        # X: list of vectors
        #    each vector is the initial representation for a sentence (more precisely, for a predicate with context)
        #    --> these are the user features in the test set
        X_reshape = X.reshape((-1, len(X)))

        # get projection matrices from trained MODEL
        user_embeddings_fromTraining = self.clf.user_embeddings
        item_embeddings_fromTraining = self.clf.item_embeddings
        
        # PREDICT
        # do the prediction for this new user via the dot product of the user feature X and the projection matrix user embeddings obtained during training
        embeddedNewUser = np.dot(X_reshape, user_embeddings_fromTraining) # now in the same space as the item embeddings obtained during training
        # use cosine similarity as similarity measure between the embedded test sentence and all the embeddings corresponding to frames
        similarity_to_all_frames = cosine_similarity(embeddedNewUser, item_embeddings_fromTraining)[0]
        
        available_frame_IDs = self.lexicon.get_available_frame_ids(lemmapos)  # get available frame IDs for this lemma.pos from lexicon
        ambig = self.lexicon.is_ambiguous(lemmapos)  # amiguous = can evoke more than one frame
        unknown = self.lexicon.is_unknown(lemmapos)  # unknown = not in lexicon

        if unknown or self.all_unknown:  # the all_unknown setting renders all lemma.pos unknown!
            available_frame_IDs = self.lexicon.get_all_frame_ids()  # if the lemma.pos is unknown, search in all frames
        else:
            # if the lemma.pos is known and has only one frame, just return it. Even if there is no data for this lemma.pos.
            if not ambig:
                return available_frame_IDs[0]
            
        # pick the best-scoring frameID among available frameIDs
        bestScore = None
        best_frame_ID = None
        for frame_ID in available_frame_IDs:
            score = similarity_to_all_frames[frame_ID]
            if bestScore is None or score >= bestScore:
                bestScore = score
                best_frame_ID = frame_ID
        return best_frame_ID
    
    
    def createInteractionMatrix(self, y_ID):
        # interactionMatrix is of size (num sentences in y_ID) x (num frames) with 1 indicating the frame label for a predicate in its context sentence
        
        numSentInY = len(y_ID)
        numFrames = len(self.lexicon.get_all_frame_ids())
        y_interactionLabels = np.zeros([numSentInY, numFrames], dtype = np.float32)
                
        for i in range(numSentInY):
            y_interactionLabels[i, y_ID[i]] = 1.
        
        return y_interactionLabels