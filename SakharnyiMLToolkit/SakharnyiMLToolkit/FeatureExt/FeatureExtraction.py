import pandas as pd
import numpy as np
import copy


class Feature_extraction:
    """
    Class representing the example class

    Attributes:
        train (str): The train data
        test (str): The test data
        matrix (list): List to store values
    """
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.matrix = []
    
    def preprocess(self):
        """
        Initialize the example class with train and test data

        Args:
            train (str): The train data
            test (str): The test data

        Returns:
            None
        """

        from itertools import product
        cols  = ["date_block_num", "shop_id", "item_id"]
        for i in range(34):
            sales = self.train[self.train.date_block_num == i]
            self.matrix.append( np.array(list( product( [i], sales.shop_id.unique(), sales.item_id.unique() ) ), dtype = np.int16) )
        
        self.matrix = pd.DataFrame( np.vstack(self.matrix), columns = cols )
        self.matrix["date_block_num"] = self.matrix["date_block_num"].astype(np.int8)
        self.matrix["shop_id"] = self.matrix["shop_id"].astype(np.int8)
        self.matrix["item_id"] = self.matrix["item_id"].astype(np.int16)
        self.matrix.sort_values( cols, inplace = True )
    
    def add_revenue(self, cols):
        """
        The method to add revenue to the data matrix.
        
        Parameters:
            cols (list): The columns to merge the matrix and the aggregated data on
        """
        self.train["revenue"] = self.train["item_cnt_day"] * self.train["item_price"]
        group = self.train.groupby( ["date_block_num", "shop_id", "item_id"] ).agg( {"item_cnt_day": ["sum"]} )
        group.columns = ["item_cnt_month"]
        group.reset_index( inplace = True)
        self.matrix = pd.merge( self.matrix, group, on = cols, how = "left" )
        self.matrix["item_cnt_month"] = self.matrix["item_cnt_month"].fillna(0).astype(np.float16)
    
    def reduce_test_memory(self):
        """
        The method to reduce memory usage of the test data.
        """
        self.test["date_block_num"] = self.test["date_block_num"].astype(np.int8)
        self.test["shop_id"] = self.test.shop_id.astype(np.int8)
        self.test["item_id"] = self.test.item_id.astype(np.int16)
    
    def concatenate_to_mx(self, cols):
        """
        This function concatenates the matrix with the droped 'ID' column from the test dataframe. The resulting dataframe is filled with 0 for any NaN values.

        Parameters:
        cols (list): List of column names to be concatenated.

        Returns:
        None
        """
        self.matrix = pd.concat([self.matrix, self.test.drop(["ID"],axis = 1)], ignore_index=True, sort=False, keys=cols)
        self.matrix.fillna( 0, inplace = True )
    
    def lag_feature(self,lags, cols):
        """
        This function lags the columns by specified number of lags. 

        Parameters:
        lags (int or list): The number of lags or a list of lags.
        cols (list): List of column names to be lagged.

        Returns:
        None
        """
        for col in cols:
            print(col)
            tmp = self.matrix[["date_block_num", "shop_id","item_id",col]]
            for i in lags:
                shifted = tmp.copy()
                shifted.columns = ["date_block_num", "shop_id", "item_id", col + "_lag_"+str(i)]
                shifted.date_block_num = shifted.date_block_num + i
                self.matrix = pd.merge(self.matrix, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    
    def add_global_item_age(self): 
        """
        This function adds the global age of each item in the matrix by computing the difference between the date_block_num and the oldest date of each item.

        Parameters:
        None

        Returns:
        None
        """ 
        oldest_date_year_month = self.matrix.groupby('item_id')['date_block_num'].min()
        merged_df = pd.merge(self.matrix, oldest_date_year_month, on='item_id')
        merged_df = merged_df.rename(columns={'date_block_num_y': 'oldest_date'})
        merged_df['item_age'] = merged_df['date_block_num_x'] - merged_df['oldest_date']
        merged_df.drop(columns=['oldest_date'], inplace=True)
        self.matrix = copy.deepcopy(merged_df)
        self.matrix.rename(columns={'date_block_num_x':'date_block_num'}, inplace=True)
        
    def add_shop_age(self):
        """
        This function adds the average sales of each item in the matrix.

        Parameters:
        nan_values (float, optional): The default value for NaN values. Defaults to 0.0.

        Returns:
        None
        """
        min_date_block_num = self.matrix.groupby('shop_id')['date_block_num'].min()
        self.matrix = pd.merge(self.matrix, min_date_block_num, on='shop_id', how='left', suffixes=('', '_min'))
        self.matrix['shop_age_in_months'] = self.matrix['date_block_num'] - self.matrix['date_block_num_min']
        self.matrix.drop(columns=['date_block_num_min'], inplace=True)
        
    def add_city_codes(self, shops):
        #name fix
        shops.loc[
            shops.shop_name == 'Сергиев Посад ТЦ "7Я"', "shop_name"
        ] = 'СергиевПосад ТЦ "7Я"'
        
        
        shops["city"] = shops["shop_name"].str.split(" ").map(lambda x: x[0])
        shops.loc[shops.city == "!Якутск", "city"] = "Якутск"
        shops["city_code"] = shops["city"].factorize()[0]
        shop_labels = shops[["shop_id", "city_code"]]
        self.matrix = self.matrix.merge(shop_labels, on='shop_id', how='left')
    
    def add_shop_type(self, shops):    
        #add shop_type as new column (str) *can better
        shops['shop_type'] = shops.apply(lambda row: 'ТЦ' if row['shop_name'].find('ТЦ')>-1\
                                else 'ТРЦ' if row['shop_name'].find('ТРЦ')>-1\
                                else 'ТРК' if row['shop_name'].find('ТРК')>-1\
                                else 'МТРЦ' if row['shop_name'].find('МТРЦ')>-1\
                                else 'ТК' if row['shop_name'].find('ТК')>-1\
                                else 'Интернет' if row['shop_name'].find('Интернет')>-1\
                                else 'Склад' if row['shop_name'].find('склад')>-1\
                                else 'Магаз', axis =1)
    
        #str -> int (mb int 16 and etc)
        shops["shop_type_code"] = shops["shop_type"].factorize()[0]
        shop_labels = shops[["shop_id", "shop_type_code"]]
        self.matrix = self.matrix.merge(shop_labels, on='shop_id', how='left')
    
    def add_item_name_groups(self, items, sim_thresh, feature_name="item_name_group"):
        """
        Adds item name groups to the matrix.

        Args:
            items (DataFrame): DataFrame of items with "item_id" and "item_name" columns.
            sim_thresh (int): Similarity threshold used to group item names.
            feature_name (str, optional): The name of the column added to the matrix for item name groups.
                Defaults to "item_name_group".

        Returns:
            None
    """
        import re
        from fuzzywuzzy import fuzz
        def partialmatchgroups(items, sim_thresh=sim_thresh):
            """
            Creates groups of item names based on similarity.

            Args:
                items (DataFrame): DataFrame of items with "item_id" and "item_name" columns.
                sim_thresh (int, optional): Similarity threshold used to group item names.

            Returns:
                DataFrame: The input DataFrame with an additional "partialmatchgroup" column for item name groups.
            """
            def strip_brackets(string):
                """
                Removes brackets from string.

                Args:
                    string (str): The input string.

                Returns:
                    str: The input string with brackets removed.
            """
                string = re.sub(r"\(.*?\)", "", string)
                string = re.sub(r"\[.*?\]", "", string)
                return string
    
            items = items.copy()
            items["nc"] = items.item_name.apply(strip_brackets)
            items["ncnext"] = np.concatenate((items["nc"].to_numpy()[1:], np.array([""])))
    
            def partialcompare(s):
                """
                Computes the partial ratio of two strings.

                Args:
                    s (Series): A row of the input DataFrame.

                Returns:
                    int: The partial ratio of the "nc" and "ncnext" columns of the input row.
                """
                return fuzz.partial_ratio(s["nc"], s["ncnext"])
    
            items["partialmatch"] = items.apply(partialcompare, axis=1)
            # Assign groups
            grp = 0
            for i in range(items.shape[0]):
                items.loc[i, "partialmatchgroup"] = grp
                if items.loc[i, "partialmatch"] < sim_thresh:
                    grp += 1
            items = items.drop(columns=["nc", "ncnext", "partialmatch"])
            return items
    
        items = partialmatchgroups(items)
        items = items.rename(columns={"partialmatchgroup": feature_name})
        items = items.drop(columns="partialmatchgroup", errors="ignore")
    
        items[feature_name] = items[feature_name].apply(str)
        items[feature_name] = items[feature_name].factorize()[0]
        self.matrix = self.matrix.merge(items[["item_id", feature_name]], on="item_id", how="left")
    
    def reduce_memory_usage(self):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = self.matrix.memory_usage().sum() / 1024**2
        for col in self.matrix.columns:
            col_type = self.matrix[col].dtypes
            if col_type in numerics:
                c_min = self.matrix[col].min()
                c_max = self.matrix[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.matrix[col] = self.matrix[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.matrix[col] = self.matrixf[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.matrix[col] = self.matrix[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self.matrix[col] = self.matrix[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        self.matrix[col] = self.matrix[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self.matrix[col] = self.matrix[col].astype(np.float32)
                    else:
                        self.matrix[col] = self.matrix[col].astype(np.float64)
        end_mem = self.matrix.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
            
            
    def sort_data(self, column_name):
        self.matrix.sort_values(by=column_name, inplace=True)
    
    def get_data(self):
        return self.matrix

    #XGBR : ['date_block_num', 'item_name_groups', 'item_price', 'average_prev_sales',
    # 'is_train', 'city_code_x', 'shop_type_code_x','item_category_id_x','platform_id',
    # 'supercategory_id']
    #CatBoostR : ['date_block_num', 'average_prev_sales', 'item_name_group_x','item_price'
    # 'is_train', 'city_code_x', 'shop_type_code_x', 'item_category_id_x', 'platform_id'
    # 'supercategory_id']
    def drop_columns(self, columns):
        self.matrix.drop(columns=columns, inplace = True)