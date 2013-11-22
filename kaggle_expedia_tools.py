import pandas as pd
from pandas.io import sql
import MySQLdb
import numpy as np
import sklearn.ensemble
import math
import pprint


#GLOBALS
DEFAULT_COLS = ['prop_log_historical_price',
                'prop_location_score1',
                'prop_location_score2',
                'loghistp_by_mean',
                'srch_room_count',
                'avg_by_price_by_mean',
                'star_by_price',
                'rev_by_price_by_mean',
                'price_by_med',
                'srch_booking_window',
                'ploc_score2_by_mean',
                'book_per_pcnt_by_mean',
                'prop_cnt',
                'promo_per_procnt',
                'train_price_avg',
                'random_bool',
                'avg_by_price',
                'srch_query_affinity_score',
                'ploc_score1_by_mean',
                'srch_length_of_stay',
                'visitor_hist_adr_usd',
                'price_usd',
                'promo_perprocnt_by_mean',
                'book_per_pcnt',
                'srch_children_count',
                'rev_by_price',
                'promotion_cnt',
                'srch_adults_count',
                'star_by_price_by_mean',
                'click_nobook_per_pcnt',
                'prop_review_score',
                'click_nobper_pcnt_by_mean']


def trainer(train_df, col_list = None, model = None, train_loc1 = 7000014, train_loc2 = 9000007, cv_loc1 = 9000008, cv_loc2 = 9917530, print_factors = True, return_model = False):
    """
    >>trainer(train_df, col_list, model = None, train_loc1 = 7000014, train_loc2 = 9000007, 
              cv_loc1 = 9000008, cv_loc2 = 9917530, print_factors = True)
    
    Given pd.DataFrame of Expedia Personalized Search training data, 
    [list of columns in DF to train on], and optionally a SciKitLearn model.  Fit the model to classify rows
    based on "click_bool" parameter (train on the rows train_loc1 to train_loc2 inclusive).  Calculate the NDCG 
    on a validation sample (rows: cv_loc1 to cv_loc2).

    Optionally print an ordered list of the column names and their "feature_importances" with print_factors.

    Optionally return_model, e.g. for use with Test data.
    """
    
    if col_list is None:
        col_list = DEFAULT_COLS


    #exclude outlying prices for training.
    train_df = train_df.loc[(train_df['price_usd'] <= 2000.0) * (train_df['price_usd'] != 0.0)]
    
    if model is None:
        model = sklearn.ensemble.RandomForestClassifier(n_estimators = 100, min_samples_split = 1000)
    
    model.fit(train_df.loc[train_loc1:train_loc2, col_list], train_df.loc[train_loc1:train_loc2, "click_bool"])
    
    feature_scores_pairs = [[model.feature_importances_[i], col_list[i]] for i in range(len(col_list))]
    
    if hasattr(model, 'predict_proba'):
        crossval_pred_arr = model.predict_proba(train_df.loc[cv_loc1:cv_loc2, col_list])[:, 1]
    else:
        crossval_pred_arr = model.predict(train_df.loc[cv_loc1:cv_loc2, col_list])

    ndcg = ndcg_calc(train_df.loc[cv_loc1:cv_loc2], crossval_pred_arr)
    
    if print_factors:
        print "NDCG:", ndcg
        print "Feature Importances:"
        pprint.pprint(sorted(feature_scores_pairs, reverse = True))
        print model
    
    if return_model:
        return model
    else:
        return ndcg

 
def ndcg_calc(train_df, pred_scores):
    """
    >>ndcg_calc(train_df, pred_scores)
       train_df: pd.DataFrame with Expedia Columns: 'srch_id', 'booking_bool', 'click_bool'
       pred_scores: np.Array like vector of scores with length = num. rows in train_df
       

    Calculate Normalized Discounted Cumulative Gain for a dataset is ranked with pred_scores (higher score = higher rank).
    If 'booking_bool' == 1 then that result gets 5 points.  If 'click_bool' == 1 then that result gets 1 point (except:
    'booking_bool' = 1 implies 'click_bool' = 1, so only award 5 points total).  
    
    NDCG = DCG / IDCG
    DCG = Sum( (2 ** points - 1) / log2(rank_in_results + 1) )
    IDCG = Maximum possible DCG given the set of bookings/clicks in the training sample.
    
    """
    eval_df = train_df[['srch_id', 'booking_bool', 'click_bool']]
    eval_df['score'] = pred_scores

    logger = lambda x: math.log(x + 1, 2)
    eval_df['log_rank'] = eval_df.groupby(by = 'srch_id')['score'].rank(ascending = False).map(logger)

    book_dcg = (eval_df['booking_bool'] * 31.0 / eval_df['log_rank']).sum() #where 2 ** 5 - 1.0 = 31.0
    book_idcg = (31.0 * eval_df['booking_bool']).sum()
    
    click_dcg = (eval_df['click_bool'] * (eval_df['booking_bool'] == 0) / eval_df['log_rank']).sum()
    
    # Max number of clicks in training set is 30.
    # Calculate the 31 different contributions to IDCG that 0 to 30 clicks have
    # and put in dict {num of click: IDCG value}.
    disc = [1.0 / math.log(i + 1, 2) if i != 0 else 0 for i in range(31)]
    disc_dict = { i: np.array(disc).cumsum()[i] for i in range(31)}
    
    # Map the number of clicks to its IDCG and subtract off any clicks due to bookings
    # since these were accounted for in book_idcg.
    click_idcg = (eval_df.groupby(by = 'srch_id')['click_bool'].sum().map(disc_dict) - eval_df.groupby(by = 'srch_id')['booking_bool'].sum()).sum()

    return (book_dcg + click_dcg) / (book_idcg + click_idcg)




def df_from_query(row_start = 7000014, row_end = 9917530, training = True, srch_start = None, srch_end = None):
    """
    >>df_from_query(row_start = 7000014, row_end = 9917530, training = True, srch_start = None, srch_end = None)
      row_start,row_end: Row number range (inclusive) to request from MySQL db (faster than srch_start,srch_end)
      srch_start,srch_end: Search ID range (inclusive) to request (slower than by row_start,row_end)
      training: BOOL, to pull from Training data set to True, set to False for Test data.
    
    Query local MySQL database, (see dbcon, below for permissions) to build a pd.DataFrame for training/testing.
    
    Several columns depend on summary statistics about Properties in the PropFactors/PropFactors7M tables in the
    database.  The "7M" suffix denotes summary statistics from the first 7000013 rows which for training
    are used with the latter ~3M rows. (This is to have these latter rows more closely resemble the test set, since some 
    properties only appear a few times so the "booking_cnt" statistic is an overly strong signal if it includes 
    information from the row that is being used for training.)  However for building the Test Set predictions, it
    is better to use statistics from the entire training set.

    Since some properties will appear in the training/test sets but not among the summary sets, columns that depend
    on these statistics are set to zero.

    Can select data by row_num or srch_id (row_num is the primary key so is faster, but it may be preferrable to 
    select by srch_id where splitting searches between chunks could cause problems, like when compiling test data).
    The presence of srch_start and srch_end overrides row_start/row_end.

    
    
    """
    
    dbcon = MySQLdb.connect('localhost', 'kaggler', 'expediapass', 'expedia')

    main_cols = "SELECT row_num, srch_id, s.prop_id, price_usd, prop_location_score1, prop_location_score2, prop_log_historical_price, prop_review_score, random_bool, srch_adults_count, srch_booking_window, srch_children_count, srch_length_of_stay, srch_query_affinity_score, srch_room_count, visitor_hist_adr_usd, (prop_starrating / price_usd) AS star_by_price, (prop_review_score / price_usd) AS rev_by_price"

    agg_cols = ", prop_cnt, promotion_cnt, train_price_avg, (booking_cnt / prop_cnt) AS book_per_pcnt, (promotion_flag / promotion_cnt) AS promo_per_procnt, (train_price_avg / price_usd) AS avg_by_price, ((click_cnt - booking_cnt) / prop_cnt) AS click_nobook_per_pcnt"

    agg_missing = ", 0 AS prop_cnt, 0 AS promotion_cnt, 0 AS train_price_avg, 0 AS book_per_pcnt, 0 AS promo_per_procnt, 0 AS avg_by_price, 0 AS click_nobook_per_pcnt"

    if training:
        from_tables1 = ", booking_bool, click_bool FROM TrainSearch AS s, PropFactors7M AS p"
        from_tables2 = ", booking_bool, click_bool FROM TrainSearch AS s"
        missing_from_table = "PropFactors7M"
    else:
        from_tables1 = " FROM TestSearch AS s, PropFactors AS p"
        from_tables2 = " FROM TestSearch AS s"
        missing_from_table = "PropFactors"

    if (srch_start is not None) or (srch_end is not None):
        if (srch_start is None):
            raise Exception("Expected selection pair (srch_start, srch_end) OR (row_start, row_end). Got: srch_start = None")
        elif (srch_end is None):
            raise Exception("Expected selection pair (srch_start, srch_end) OR (row_start, row_end). Got: srch_end = None")
        else: 
            where_str = " AND srch_id >= " + str(srch_start) + " AND srch_id <= " + str(srch_end) + ";"
    else:
        where_str = " AND row_num >= " + str(row_start) + " AND row_num <= "+ str(row_end) + ";"

    train_dfmost = sql.frame_query(main_cols + agg_cols + from_tables1 + " WHERE p.prop_id = s.prop_id" + where_str, con = dbcon, index_col = 'row_num')
    
    train_dfpropmissing = sql.frame_query(main_cols + agg_missing + from_tables2 + " WHERE prop_id NOT IN (SELECT prop_id FROM " + missing_from_table + ")" + where_str, con = dbcon, index_col = 'row_num')
    
    dbcon.close()

    train_df = pd.concat([train_dfmost, train_dfpropmissing])
    train_df.sort(inplace = True)
    
    
    train_df.fillna(value = 0, inplace = True)

    return train_df



def agg_by_srch(train_df):
    """
    >>agg_by_srch(train_df)
    
    Given a training/testing dataset that includes the columns listed in cols_in below, calculate
    new columns that normalize these values among the search in which they appear by dividing by
    the mean (or median for 'price_usd' since it has higher variance).  i.e. given prop_location_score "j"
    for search "i" add column: prop_location_score[i][j] / mean(prop_location_score[i])

    """
    
    cols_in = ['prop_location_score1', 
               'prop_location_score2', 
               'book_per_pcnt', 
               'avg_by_price', 
               'rev_by_price', 
               'star_by_price', 
               'promo_per_procnt',
               'prop_log_historical_price', 
               'click_nobook_per_pcnt']

    cols_out = ['srch_ploc_score1_mean', 
                'srch_ploc_score2_mean', 
                'srch_book_per_pcnt_mean', 
                'srch_avg_by_price_mean', 
                'srch_rev_by_price_mean', 
                'srch_star_by_price_mean', 
                'srch_promo_per_procnt_mean', 
                'srch_prop_log_historical_price_mean', 
                'click_nobook_per_pcnt_mean']

    srch_df = train_df[['srch_id'] + cols_in].groupby(by='srch_id').mean()
    srch_df.columns = cols_out

    srch_df['srch_price_med'] = train_df[['srch_id','price_usd']].groupby(by = 'srch_id').median()
    srch_df['srch_id'] = srch_df.index
    
    cols_inp = list(cols_in)
    cols_outp = list(cols_out)
    cols_inp.append('price_usd')
    cols_outp.append('srch_price_med')

    cols_agg = ['ploc_score1_by_mean', 
                'ploc_score2_by_mean', 
                'book_per_pcnt_by_mean', 
                'avg_by_price_by_mean', 
                'rev_by_price_by_mean', 
                'star_by_price_by_mean', 
                'promo_perprocnt_by_mean', 
                'loghistp_by_mean', 
                'click_nobper_pcnt_by_mean',
                'price_by_med']

    for i in xrange(len(cols_inp)):
        train_df[cols_agg[i]] = train_df[cols_inp[i]] / train_df.join(srch_df[['srch_id', cols_outp[i]]], on='srch_id', rsuffix = '_agg')[cols_outp[i]]
    
    train_df.fillna(value = 0, inplace = True)
    train_df.loc[train_df.price_by_med == np.inf, 'price_by_med'] = 0

    return train_df
    


def test_pred_sorted(test_df, model, cols = None, regress_model = False):
    """
    >>test_pred_sorted(test_df, model, cols, regress_model = False)
        test_df: pd.DataFrame that contains columns listed in cols.
        model: SciKitLearn model used for ranking.
        cols: [list of strs] columns in test_df to send to model.
        regress: BOOL, Regressor Model used, as opposed to Classifier.
    
    Return a pd.DataFrame that contains 'srch_id', and 'property_id' columns such that
    the properties are listed in descending order of their model score within each search.
    
    To save output use: test_out_df.to_csv("FILE OUT", index = False, cols = ['srch_id', 'prop_id'], header = ['SearchId','PropertyId'])
    """
    if cols is None:
        cols = DEFAULT_COLS

    if regress_model:
        scores = model.predict(test_df[cols])
    else:
        scores = model.predict_proba(test_df[cols])[:, 1]
    test_df['sort_score'] = scores
     
    return test_df[['srch_id', 'prop_id', 'sort_score']].sort(columns=['srch_id', 'sort_score'], ascending = [True, False])


def training_loader(row_start = 7000014, row_end = 9917530):
    """
    >>training_loader(row_start = 7000014, row_end = 9917530)

    Calls df_from_query() to retrieve chunks of training data ~1M rows
    at a time.  Returns these as a single DataFrame.
    """
    num_calls = 1 + (row_end - row_start) // 1000000

    chunk_list = []
    for i in xrange(num_calls):
        rs = row_start + i * 1000000
        re = min(rs + 1000000, row_end)

        chunk_df = df_from_query(row_start = rs, row_end = re, training = True)
        chunk_list.append(chunk_df)

    return pd.concat(chunk_list)

def test_loader_saver(model, cols = None, file_str = "test_preds_saved", regress_model = False):
    """
    >>test_loader_saver(model, cols, file_str = "test_preds_saved", regress_model = False)
        model: SciKitLearn model used for ranking.
        cols: [list of strs] columns in test_df to send to model.
        file_str: STRING, file name of test predictions to be saved.
        regress_model: BOOL, Regressor Model used, as opposed to Classifier.
    
    Trying to run the MySQL query to load the entire Test set was too memory intensive for my 4Gb machine, 
    so load and process the data in chunks.

    Then join the files together on the command line:
    cat test_preds_saved0.csv test_preds_saved1.csv test_preds_saved2.csv test_preds_saved3.csv test_preds_saved4.csv test_preds_saved5.csv > test_preds_all_ADDHEADER.csv
    """
    
    if cols is None:
        cols = DEFAULT_COLS

    starts = [2, 110001, 220001, 330001, 440001, 550001]
    ends = [110000, 220000, 330000, 440000, 550000, 700000]

    for i in xrange(6):
        test_df = df_from_query(srch_start = starts[i], srch_end = ends[i], training = False)
        agg_by_srch(test_df)
        
        if regress_model:
            pred_df = test_pred_sorted(test_df, model, cols = cols, regress_model = True)
        else:
            pred_df = test_pred_sorted(test_df, model, cols = cols, regress_model = False)

        if i == 0:
            pred_df.to_csv(file_str + str(i) + ".csv", index = False, cols = ['srch_id', 'prop_id'], header = ['Search_Id', 'PropertyId'])
        else:
            pred_df.to_csv(file_str + str(i) + ".csv", index = False, cols = ['srch_id', 'prop_id'], header = False)
        
