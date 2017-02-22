#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Jun 10, 2016

.. codeauthor: svitlana vakulenko <svitlana.vakulenko@gmail.com>
'''
from twython import Twython, TwythonRateLimitError, TwythonError
from twitter_settings import *
import json
from datetime import datetime
import pause

SNOW_TEST = 'snow14_testset_ids.txt'
SNOW_FOLDER = './'
SNOW_LOG = 'snow_progress.log'


class TwitterIDsDownloader():

    def __init__(self, ids_file, response_file=None,
                 log_file='ids_downloaded.log'):
        self.ids_file = ids_file
        if response_file:
            self.response_file = response_file
        else:
            self.response_folder = '/tweets/'
            self.response_file = self.get_new_response_file()
        self.log_file = log_file
        self.batch_size = 100
        self.tweet_ids = []
        self.get_current_status()
        self.time_outs = 0
        self.split_every = 5

    def get_new_response_file(self):
        return str(datetime.now())

    def get_current_status(self):
        with open(self.log_file, 'w+') as log, open(self.ids_file) as fr:
            processed_ids = set(log.read().splitlines())
            print str(len(processed_ids)) + " ids processed"
            ids = set(fr.read().splitlines())
            # exclude already processed ids
            ids.difference_update(processed_ids)
            print str(len(ids)) + " ids to process"
            self.tweet_ids = list(ids)

    def call_API(self, ids):
        ids_string = ','.join(ids)
        # establish loop to achieve the successfulr retrieval
        success = False
        while not success:
            try:
                twitter_client = Twython(APP_KEY, APP_SECRET,
                                         OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
                tweets = twitter_client.lookup_status(id=ids_string)
                # return tweets
                if tweets:
                    with open(self.response_file, 'a+') as fw:
                        fw.write(json.dumps(tweets) + '\n')
                # save processed ids to file
                with open(self.log_file, 'a') as log:
                    log.writelines([tweet_id + '\n' for tweet_id in ids])
                print str(len(tweets)) + " tweets returned from the API"
                success = True
            except TwythonRateLimitError as e:
                unix_due_time = float(e.retry_after)
                due_time = datetime.fromtimestamp(unix_due_time)
                print "\nSleeping until " + due_time.strftime('%H:%M')
                pause.until(unix_due_time)
                self.time_outs += 1
                # write into a new file
                if self.time_outs % self.split_every == 0:
                    self.response_file = self.get_new_response_file()
                # or 15 mins time-out
                # time.sleep(60 * 15)
            except TwythonError as e:
                print e
                pass

    def download_tweets(self):
        if self.tweet_ids:
            if len(self.tweet_ids) < self.batch_size:
                self.call_API(self.tweet_ids)
            else:
                for batch in zip(* [iter(self.tweet_ids)] * self.batch_size):
                    self.call_API(list(batch))
        else:
            print "No tweets to download"


def test_download_tweets():
    ids_file = 'test_tweet_ids_200.txt'
    response_file = 'test_tweets.json'
    log_file = 'test_processed_ids.log'
    processor = TwitterIDsDownloader(ids_file, response_file, log_file)
    processor.download_tweets()


def download_SNOW_tweets():
    ids_file = SNOW_FOLDER + SNOW_TEST
    processor = TwitterIDsDownloader(ids_file, log_file=SNOW_LOG)
    processor.download_tweets()


if __name__ == '__main__':
    test_download_tweets()
    # first download the SNOW ids from https://figshare.com/articles/SNOW_2014_Data_Challenge/1003755
    download_SNOW_tweets()
