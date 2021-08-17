import gym
from gym import spaces
import string
from gym.spaces.multi_discrete import MultiDiscrete
import numpy as np
from gym.utils import seeding
import random
import collections
from sklearn.feature_extraction.text import CountVectorizer


class HangmanEnv(gym.Env):
    
    def __init__(self):
        # super().__init__()
        
        self.mistakes_done = 0
        self.action_space = spaces.Discrete(26)
        f = open("./words.txt", 'r').readlines()
        self.wordlist = [w.strip() for w in f]
        self.vectorizer = CountVectorizer(tokenizer=lambda x: list(x))
        self.vectorizer2 = CountVectorizer(tokenizer=lambda x: list(x))
        self.vectorizer2.fit([string.ascii_lowercase])
        self.vectorizer.fit([string.ascii_lowercase, "_"])
        self.observation_space = spaces.Tuple((
            spaces.MultiDiscrete(np.array([25]*27)),
            spaces.MultiDiscrete(np.array([1]*26)),
        ))
        self.observation_space.shape=(27, 26)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def choose_word(self):
        return random.choice(self.wordlist)
    
    def count_words(self, word):
        lens = [len(w) for w in self.wordlist]
        counter=dict(collections.Counter(lens))
        return counter[len(word)]
    
    def get_current_worddict(self):
        return np.array([1 if w in self.word_subset else 0 for w in self.wordlist])
        
    def reset(self):
        self.mistakes_done = 0
        self.word = self.choose_word()
        self.wordlen = len(self.word)
        self.gameover = False
        self.win = False
        self.guess_string = "_"*self.wordlen
        self.actions_used = []
        self.actions_correct = []
        
        print("Word Selected = ", self.word)
        
        self.state = (
            self.vectorizer.transform([self.guess_string]).toarray()[0],
            np.array([0]*26),       
        )
        
        print("State = ", self.state)
        return self.state
        
    def vec2letter(self, action):
        letters = string.ascii_lowercase
        # idx = np.argmax(action==1)
        return letters[action]
    
    def getGuessedWord(self, secretWord, lettersGuessed):
        """
        secretWord: string, the word the user is guessing
        lettersGuessed: list, what letters have been guessed so far
        returns: string, comprised of letters and underscores that represents
        what letters in secretWord have been guessed so far.
        """
        secretList = []
        secretString = ''
        for letter in secretWord:
            secretList.append(letter)
        for letter in secretList:
            if letter not in lettersGuessed:
                letter = '_'
            secretString += letter  
        return secretString
    
    
    def check_guess(self, letter):
        if letter in self.word:
            self.prev_string = self.guess_string
            self.actions_correct.append(letter)
            self.guess_string = self.getGuessedWord(self.word, self.actions_correct)
            return True
        else:
            return False
        
    # def findOccurrences(self, s, ch):
    #     return [i for i, letter in enumerate(s) if letter == ch]
        
    # def evaluate_subset(self, action):
        
    #     self.word_subset = [w for w in self.word_subset if self.findOccurrences(self.word, action) == self.findOccurrences(w, action)]
    #     self.curr_num = len(self.word_subset)
    #     print("subset length = ", self.num_words)
    #     print("curr selected =", self.curr_num)
    
    def step(self, action):
        done = False
        reward = 0
        if string.ascii_lowercase[action] in self.actions_used:
            reward = -2
            self.mistakes_done += 1
        elif string.ascii_lowercase[action] in self.actions_correct:
            reward = -5
        elif self.check_guess(self.vec2letter(action)):
            print("Correct guess, evaluating reward")
            if(set(self.word) == set(self.actions_correct)):
                reward = 10
                done = True
                self.win = True
                self.gameover = True
            
            # self.evaluate_subset(action)
            reward = +1
            # reward = self.edit_distance(self.state, self.prev_string)
            self.actions_correct.append(string.ascii_lowercase[action])
        else:
            self.mistakes_done += 1
            if(self.mistakes_done >= 6):
                reward = -5
                done = True
                self.gameover = True
            else:
                reward = -2
        if string.ascii_lowercase[action] not in self.actions_used:
            self.actions_used.append(string.ascii_lowercase[action])
        
        self.state = (
            self.vectorizer.transform([self.guess_string]).toarray()[0],
            self.vectorizer2.transform(self.actions_used).toarray()[0] 
        )
        return (self.state, reward, done, {'win' :self.win, 'gameover':self.gameover})
    
    
class HangmanPlayer():
    pass