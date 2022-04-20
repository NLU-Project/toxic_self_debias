import pandas as pd
import requests
import os
import json
import argparse
import time


# To set your environment variables in your terminal run the following line:
# os.environ['BEARER_TOKEN'] = <<YOUR TOKEN>>>
bearer_token = os.environ.get("BEARER_TOKEN")

def create_url(user_id):
    # Replace with user ID below
    return "https://api.twitter.com/2/users/{}/tweets".format(user_id)

def get_params(max_results):
    # Tweet fields are adjustable.
    # Options include:
    # attachments, author_id, context_annotations,
    # conversation_id, created_at, entities, geo, id,
    # in_reply_to_user_id, lang, non_public_metrics, organic_metrics,
    # possibly_sensitive, promoted_metrics, public_metrics, referenced_tweets,
    # source, text, and withheld
    return {"tweet.fields": "author_id,created_at,lang,possibly_sensitive,id,text",
            'max_results':max_results}


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2UserTweetsPython"
    return r


def connect_to_endpoint(url, params):
    response = requests.request("GET", url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()

def fetch_json(user_id,max_results):
    url = create_url(user_id)
    params = get_params(max_results)
    json_response = connect_to_endpoint(url, params)
    return(json_response)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_data", type=str, required = True,
                        help = "twitter user file in data directory")
    parser.add_argument("--data_dir", type = str, required = True,
                        help = "Based directory for reading in and writing out datasets")
    parser.add_argument("--tweet_out", type=str, required = True,
                        help = "where to write user's collected tweets")
    parser.add_argument("--error_out", type=str, required = True,
                        help = "where to write user's errors when collecting tweets")
    parser.add_argument("--max_results", type = int, required = True,
                        help = "max results per user")
    parser.add_argument("--max_users", type = int, required = True,
                        help = "max results per user")                        
    args = parser.parse_args()


    #merge data
    data_df_list = []
    error_df_list = []

    #read user data df
    user_df_filepath = os.path.join(args.data_dir,args.user_data)
    users_df = pd.read_csv(user_df_filepath)
    print('reading user data from {}'.format(user_df_filepath))


    for index, row in users_df.iterrows():
      print('Index: {0} -- Collecting Tweets from User Id: {1}'.format(index,row.user_id))
      if index > args.max_users:
          print('max users reached')
          break

      try:
        data_dict = fetch_json(int(row.user_id),args.max_results)
      except:
          print('unable to fetch json for user id {}'.format(row.user_id))
          print('going to sleep')
          time.sleep(60*15+5)
          try:
            print('trying to fetch {} again after wait period'.format(row.user_id))
            data_dict = fetch_json(int(row.user_id),args.max_results)
          except:
            print('unable to fetch again, skipping {}'.format(row.user_id))
            continue        

      if 'data' in data_dict:
        data_df = pd.DataFrame.from_dict(data_dict['data'])
        
        data_df['user_id'] = row.user_id
        data_df['is_female'] = row.is_female
        data_df['year_born'] = row.year_born
        data_df['race'] = row.race

        data_df_list.append(data_df)

      elif 'errors' in data_dict:
        print('Error collecting Tweets from user {}'.format(row.user_id))
        error_df = pd.DataFrame.from_dict(data_dict['errors'])

        error_df['is_female'] = row.is_female
        error_df['year_born'] = row.year_born
        error_df['race'] = row.race

        error_df_list.append(error_df)

    user_tweet_df = pd.concat(data_df_list).reset_index(drop = True)
    user_error_df = pd.concat(error_df_list).reset_index(drop = True)

    tweet_out_filepath = os.path.join(args.data_dir,"{0}-ids_{1}-tweets_".format(args.max_users, args.max_results) + args.tweet_out)
    error_out_filepath = os.path.join(args.data_dir,"{0}-ids_{1}-tweets_".format(args.max_users, args.max_results) + args.error_out)

    user_tweet_df.to_csv(tweet_out_filepath)
    user_error_df.to_csv(error_out_filepath)

    print('COMPLETE')

    #test for reading in chell command
    print (args.max_results)
    print (user_df_filepath)
    print (tweet_out_filepath)
    print (error_out_filepath)


if __name__ == '__main__':
    main()