package scalation
package modeling


import scalation.mathstat._
import scala.math.{sqrt}
import scalation.mathstat.Scala2LaTeX._


def InSample (ox: MatrixD,  y: VectorD, ox_fname: Array[String], data_name: String): Unit =

    val n_q = 15
    val r_q = 0 until 15
    val colName = "Metric, Regression, Ridge, Lasso, Sqrt, log1p"
    val rowName = modeling.qoF_names.take(n_q)  

    val reg = new Regression (ox, y, ox_fname)
    val qof1 = reg.trainNtest ()()._2(r_q) // train and test on the same data (in-sample)

    val ox_centered = ox - ox.mean(0) // center the data by subtracting the mean of each column
    val y_centered  = y - y.mean         // center the target variable

    val ridge = new RidgeRegression (ox_centered, y_centered, ox_fname)
    val qof2 = ridge.trainNtest ()()._2(r_q) // train and test on the same data (in-sample)

    val lasso = new LassoRegression (ox, y, ox_fname)
    val qof3 = lasso.trainNtest ()()._2(r_q) // train and test on the same data (in-sample)

    val Sqrt = new TranRegression (ox, y, ox_fname, tran=sqrt, itran=sq)
    val qof4 = Sqrt.trainNtest ()()._2(r_q) // train and test on the same data (in-sample)

    val Log1p = new TranRegression (ox, y, ox_fname)
    val qof5 = Log1p.trainNtest ()()._2(r_q) // train and test on the same data (in-sample)

    
    val caption = data_name + " In-Sample QoF Comparison"
    val name    = "tab:" + caption
    val qof_matrix = MatrixD (qof1, qof2, qof3, qof4, qof5).transpose // transpose to get metrics as rows and models as columns

    val latex   = make_doc (make_table (caption, name, qof_matrix, colName, rowName))
    println (latex)

end InSample

@main def P1_AutoMPG_IS (): Unit =
    import Example_AutoMPG._
    InSample (ox, y, ox_fname, "Auto MPG")
end P1_AutoMPG_IS

@main def P1_Housing_IS (): Unit =
    val ox_fname = Array ("intercept", "Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality")
    val xy = MatrixD.load ("house_price_regression_dataset.csv", 1, sp=',')      // load the dataset, skipping the header row
    val x = xy(?, 0 until 6)                                  // get the first 6 columns as the feature matrix
    val y = xy(?, 7)                                        // get the 7th column as the response vector
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones    
    val ox     = _1 +^: x                                              // prepend a column of all ones to x
    InSample (ox, y, ox_fname, "Housing Prices")
end P1_Housing_IS

@main def P1_Insurance_IS (): Unit =
    val ox_fname = Array ("intercept", "age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest")
    val oxy = MatrixD.load ("insurance_cat2num.csv", 1, sp=',')      // load the dataset, skipping the header row
    val ox = oxy(?, 0 until 8)                                  // get the first 8 columns as the feature matrix
    val y = oxy(?, 9)                                        // get the 9th column as the response vector
    InSample(ox, y, ox_fname, "Insurance Charges")
end P1_Insurance_IS

def OutOfSample (ox: MatrixD,  y: VectorD, ox_fname: Array[String], data_name: String, testp: Double): Unit =

    val n_q = 15
    val r_q = 0 until 15
    val colName = "Metric, Regression, Ridge, Lasso, Sqrt, log1p"
    val rowName = modeling.qoF_names.take(n_q)

    val permGen = scalation.mathstat.TnT_Split.makePermGen (ox.dim)             // make a permutation generator
    val n_test = (ox.dim * testp).toInt                                           // (1-testp)% training, testp% testing
    val idx = scalation.mathstat.TnT_Split.testIndices(permGen, n_test)         // get test indices for 80-20 split
    val (ox_test, ox_train) = TnT_Split (ox, idx)                               // TnT split the dataset ox (row split)
    val yy = MatrixD.fromVector (y)                                // turn the m-vector y into an m-by-1 matrix 
    val (yy_test, yy_train) = TnT_Split (yy, idx)                               // TnT split the response vector y (row split)
    val y_train = yy_train.col(0)                                                     // get the test response vector from the test response matrix
    val y_test = yy_test.col(0)                                                       // get the test response vector from the test response matrix
    

    val reg = new Regression (ox, y, ox_fname)
    val qof1 = reg.trainNtest (ox_train, y_train)(ox_test, y_test)._2(r_q) // train and test on the same data (in-sample)

    val ox_centered = ox - ox_train.mean(0) // center the data by subtracting the mean of each column
    val y_centered  = y - y_train.mean         // center the target variable
    val ox_train_centered = ox_train - ox_train.mean(0) // center the data by subtracting the mean of each column
    val y_train_centered  = y_train - y_train.mean         // center the target variable
    val ox_test_centered = ox_test - ox_train.mean(0) // center the data by subtracting the mean of each column
    val y_test_centered  = y_test - y_train.mean         // center the target variable

    val ridge = new RidgeRegression (ox_centered, y_centered, ox_fname)
    val qof2 = ridge.trainNtest (ox_train_centered, y_train_centered)(ox_test_centered, y_test_centered)._2(r_q) // train and test on the same data (in-sample)

    val lasso = new LassoRegression (ox, y, ox_fname)
    val qof3 = lasso.trainNtest (ox_train, y_train)(ox_test, y_test)._2(r_q) // train and test on the same data (in-sample)

    val Sqrt = new TranRegression (ox, y, ox_fname, tran=sqrt, itran=sq)
    // val qof4 = Sqrt.trainNtest (ox_train, y_train)(ox_test, y_test)._2(r_q) // train and test on the same data (in-sample)
    val (sqrtyp, qof4_full) = Sqrt.validate ()() // train on the training data and validate on the test data (validation)
    val qof4 = qof4_full(r_q) // get the QoF metrics for the core metrics only (r_q)
    val (y_test_ord_1, sqrtyp_ord) = orderByY (y_test, sqrtyp) // order the test response and the sqrt predictions by the test response for better visualization
    new Plot (null, y_test_ord_1, sqrtyp_ord, s"Plot ${Sqrt.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)

    val Log1p = new TranRegression (ox, y, ox_fname)
    // val qof5 = Log1p.trainNtest (ox_train, y_train)(ox_test, y_test)._2(r_q) // train and test on the same data (in-sample)
    val (log1pyp, qof5_full) = Log1p.validate ()() // train on the training data and validate on the test data (validation)
    val qof5 = qof5_full(r_q) // get the QoF metrics for the core metrics only (r_q)
    val (y_test_ord_2, log1yp_ord) = orderByY (y_test, log1pyp) // order the test response and the log1p predictions by the test response for better visualization
    new Plot (null, y_test_ord_2, log1yp_ord, s"Plot ${Log1p.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    
    val caption = data_name + " Out-of-Sample QoF Comparison"
    val name    = "tab:" + caption
    val qof_matrix = MatrixD (qof1, qof2, qof3, qof4, qof5).transpose // transpose to get metrics as rows and models as columns

    val latex   = make_doc (make_table (caption, name, qof_matrix, colName, rowName))
    println (latex)

end OutOfSample

@main def P1_AutoMPG_OOS (): Unit =
    import Example_AutoMPG._
    OutOfSample (ox, y, ox_fname, "Auto MPG", 0.2)
end P1_AutoMPG_OOS

@main def P1_Housing_OOS (): Unit =
    val ox_fname = Array ("intercept", "Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality")
    val xy = MatrixD.load ("house_price_regression_dataset.csv", 1, sp=',')      // load the dataset, skipping the header row
    val x = xy(?, 0 until 6)                                  // get the first 6 columns as the feature matrix
    val y = xy(?, 7)                                        // get the 7th column as the response vector
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones    
    val ox     = _1 +^: x                                              // prepend a column of all ones to x
    OutOfSample (ox, y, ox_fname, "Housing Prices", 0.2)
end P1_Housing_OOS

@main def P1_Insurance_OOS (): Unit =
    val ox_fname = Array ("intercept", "age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest")
    val oxy = MatrixD.load ("insurance_cat2num.csv", 1, sp=',')      // load the dataset, skipping the header row
    val ox = oxy(?, 0 until 8)                                  // get the first 8 columns as the feature matrix
    val y = oxy(?, 9)                                        // get the 9th column as the response vector
    OutOfSample(ox, y, ox_fname, "Insurance Charges", 0.2)
end P1_Insurance_OOS