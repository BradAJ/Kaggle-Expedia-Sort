import pandas as pd
import numpy as np
import sklearn.ensemble
import sklearn.tree
import math
import pprint
import MySQLdb
from pandas.io import sql
import cPickle as pkl


def trainer(train_df, col_list, model = None, train_loc1 = 1, train_loc2 = 100020, cv_loc1 = 100021, cv_loc2 = 200009, print_factors = True):
    
    #exclude outlying prices for training.
    train_df = train_df.loc[(train_df['price_usd'] <= 2000.0) * (train_df['price_usd'] != 0.0)]
    
    if model is None:
        model = sklearn.ensemble.RandomForestClassifier(n_estimators = 100, min_samples_split = 1000)
    
    model.fit(train_df.loc[train_loc1:train_loc2, col_list], train_df.loc[train_loc1:train_loc2, "booking_bool"])
    
    feature_scores_pairs = [[model.feature_importances_[i], col_list[i]] for i in range(len(col_list))]
    
    if hasattr(model, 'predict_proba'):
        crossval_pred_arr = model.predict_proba(train_df.loc[cv_loc1:cv_loc2, col_list])[:, 1]
    else:
        crossval_pred_arr = model.predict(train_df.loc[cv_loc1:cv_loc2, col_list])

    ndcg = ndcg_calc(train_df.loc[cv_loc1:cv_loc2], crossval_pred_arr)
    
    if print_factors:
        print ndcg
        pprint.pprint(sorted(feature_scores_pairs, reverse = True))
        print model
    
    if return_model:
        return model, feature_scores_pairs
    else:
        return ndcg


def df_from_query(col_list = None, row_limit = 200009, fill_na_0 = True, books_only = False):
    """
    Return a dataframe by calling the local Expedia MySQL database (Tables: PropFactors, UserSrchUnique, SrchUniversals) 
    and returning a dataframe with the columns of interest in col_list (or a default set) along with ID and result columns: 
    row_num, srch_id, booking_bool, click_bool.
    The row_limit the rows are returned up to and including row_num = row_limit (where 2000009 cuts between searches 13431 and 13432).
    Set row_limit = 9917530 to include the entire training sample.

    Some of the combo columns may have div by 0 so convert NaN -> 0 by default.

    big_query = "SELECT row_num, u.srch_id, booking_bool, click_bool, orig_destination_distance, price_usd, promotion_flag, prop_brand_bool, prop_country_id, prop_location_score1, prop_location_score2, prop_log_historical_price, prop_review_score, prop_starrating, random_bool, site_id, srch_adults_count, srch_booking_window, srch_children_count, srch_destination_id, srch_length_of_stay, srch_query_affinity_score, srch_room_count, srch_saturday_night_bool, visitor_hist_adr_usd, visitor_hist_starrating, visitor_location_country_id, prop_cnt, booking_cnt, click_cnt, promotion_cnt, train_price_avg, p.prop_id, (booking_cnt / prop_cnt) AS book_per_pcnt, (click_cnt / prop_cnt) AS click_per_pcnt, (promotion_flag / promotion_cnt) AS promo_per_procnt, ((booking_cnt - click_cnt) / prop_cnt) AS click_nobook_per_pcnt, (prop_review_score / price_usd) AS rev_by_price, (prop_starrating / price_usd) AS star_by_price, (train_price_avg / price_usd) AS avg_by_price FROM UserSrchUnique AS s, PropFactors7M AS p, SrchUniversals AS u WHERE p.prop_id = s.prop_id AND s.srch_id = u.srch_id AND row_num <=100019;"
    """


    if col_list == None:
        def_q = ', '.join(default_cols)
        agg_q = ', '.join(agg_cols)
        query_cols = def_q +', '+ agg_q + ', ' + combo_cols_query
    else:
        query_cols = ', '.join(col_list) + ', ' + combo_cols_query
    
    if books_only == True:
        query_str = 'SELECT row_num, u.srch_id, booking_bool, click_bool, ' + query_cols + ' FROM UserSrchUnique AS s, PropFactors7M AS p, SrchUniversals AS u WHERE p.prop_id = s.prop_id AND s.srch_id = u.srch_id AND s.booking_bool = 1 AND row_num <=' + str(row_limit) + ';'
    else:
        query_str = 'SELECT row_num, u.srch_id, booking_bool, click_bool, ' + query_cols + ' FROM UserSrchUnique AS s, PropFactors7M AS p, SrchUniversals AS u WHERE p.prop_id = s.prop_id AND s.srch_id = u.srch_id AND row_num <=' + str(row_limit) + ';'

    print query_str
    dbcon = MySQLdb.connect('localhost', 'kaggler', 'expediapass', 'expedia')

    query_df = sql.frame_query(query_str, con = dbcon, index_col = 'row_num')
    dbcon.close()

    if fill_na_0:
        query_df = query_df.fillna(value = 0)
    return query_df
 
def ndcg_calc(train_df, pred_scores):
    """
    Calculate Normalized Discounted Cumulative Gain for a dataset is ranked with pred_scores (higher score = higher rank).
    If 'booking_bool' == 1 then that result gets 5 points.  If 'click_bool' == 1 then that result gets 1 point (except:
    'booking_bool' = 1 implies 'click_bool' = 1, so only award 5 points total).  
    
    DCG = Sum( (2 ** points - 1) / log2(rank_in_results + 1) )
    The numerator is 31 for a booking and 1 for a click.
    IDCG = Maximum possible DCG given the set of bookings/clicks in the training sample.
    
    """
    eval_df = train_df[['srch_id', 'booking_bool', 'click_bool']]
    eval_df['score'] = pred_scores

    logger = lambda x: math.log(x + 1, 2)
    eval_df['log_rank'] = eval_df.groupby(by = 'srch_id')['score'].rank(ascending = False).map(logger)

    book_dcg = (eval_df['booking_bool'] * 31.0 / eval_df['log_rank']).sum()
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









def agg_by_srch_v2(train_df, cols = None):
    cols_in = ['prop_location_score1', 'prop_location_score2', 'book_per_pcnt', 'avg_by_price', 'rev_by_price', 'star_by_price', 'promo_per_procnt', 'prop_log_historical_price', 'click_nobook_per_pcnt']
    srch_df = train_df[['srch_id'] + cols_in].groupby(by='srch_id').mean()
    cols_out = ['srch_ploc_score1_mean', 'srch_ploc_score2_mean', 'srch_book_per_pcnt_mean', 'srch_avg_by_price_mean', 'srch_rev_by_price_mean', 'srch_star_by_price_mean', 'srch_promo_per_procnt_mean', 'srch_prop_log_historical_price_mean', 'click_nobook_per_pcnt_mean']
    srch_df.columns = cols_out

    srch_df['srch_price_med'] = train_df[['srch_id','price_usd']].groupby(by = 'srch_id').median()
    srch_df['srch_id'] = srch_df.index
    
    cols_inp = list(cols_in)
    cols_outp = list(cols_out)
    cols_inp.append('price_usd')
    cols_outp.append('srch_price_med')

    cols_agg = ['ploc_score1_by_mean', 'ploc_score2_by_mean', 'book_per_pcnt_by_mean', 'avg_by_price_by_mean', 'rev_by_price_by_mean', 'star_by_price_by_mean', 'promo_perprocnt_by_mean', 'loghistp_by_mean', 'click_nobper_pcnt_by_mean', 'price_by_med']

    for i in xrange(len(cols_inp)):
        train_df[cols_agg[i]] = train_df[cols_inp[i]] / train_df.join(srch_df[['srch_id', cols_outp[i]]], on='srch_id', rsuffix = '_agg')[cols_outp[i]]
    
    train_df.fillna(value = 0, inplace = True)
    

    if cols != None:
        cols_ids = set(cols)
        cols_ids.add('srch_id')
        cols_ids.add('prop_id')
        cols_ids.add('booking_bool')
        cols_ids.add('click_bool')
        return train_df[list(cols_ids)]
    else:
        return train_df

def agg_by_srch_v3(train_df, cols = None):
    cols_in = ['prop_location_score1', 'prop_location_score2', 'book_per_pcnt', 'avg_by_price', 'rev_by_price', 'star_by_price', 'promo_per_procnt', 'prop_log_historical_price', 'click_nobook_per_pcnt', 'ploc_score1_by_price', 'ploc_score2_by_price', 'srch_query_affinity_score']
    srch_df = train_df[['srch_id'] + cols_in].groupby(by='srch_id').mean()
    cols_out = ['srch_ploc_score1_mean', 'srch_ploc_score2_mean', 'srch_book_per_pcnt_mean', 'srch_avg_by_price_mean', 'srch_rev_by_price_mean', 'srch_star_by_price_mean', 'srch_promo_per_procnt_mean', 'srch_prop_log_historical_price_mean', 'click_nobook_per_pcnt_mean', 'ploc_score1_mean', 'ploc_score2_mean', 'affinity_score_mean']
    srch_df.columns = cols_out

    srch_df['srch_price_med'] = train_df[['srch_id','price_usd']].groupby(by = 'srch_id').median()
    srch_df['srch_id'] = srch_df.index
    
    cols_inp = list(cols_in)
    cols_outp = list(cols_out)
    cols_inp.append('price_usd')
    cols_outp.append('srch_price_med')

    cols_agg = ['ploc_score1_by_mean', 'ploc_score2_by_mean', 'book_per_pcnt_by_mean', 'avg_by_price_by_mean', 'rev_by_price_by_mean', 'star_by_price_by_mean', 'promo_perprocnt_by_mean', 'loghistp_by_mean', 'click_nobper_pcnt_by_mean', 'ploc_score1_by_price_by_mean', 'ploc_score2_by_price_by_mean', 'affinity_score_by_mean', 'price_by_med']

    for i in xrange(len(cols_inp)):
        train_df[cols_agg[i]] = train_df[cols_inp[i]] / train_df.join(srch_df[['srch_id', cols_outp[i]]], on='srch_id', rsuffix = '_agg')[cols_outp[i]]
    
    train_df.fillna(value = 0, inplace = True)
    

    if cols != None:
        cols_ids = set(cols)
        cols_ids.add('srch_id')
        cols_ids.add('prop_id')
        cols_ids.add('booking_bool')
        cols_ids.add('click_bool')
        return train_df[list(cols_ids)]
    else:
        return # train_df
