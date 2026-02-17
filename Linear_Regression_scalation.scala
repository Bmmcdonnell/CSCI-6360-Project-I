package scalation
package modeling

// import scalation.mathstat._
import scalation.modeling._

@main def LinRegAutoMPG (): Unit =

    import Example_AutoMPG._
    import scalation.mathstat._
    import scalation.mathstat.Scala2LaTeX._
    // import scala.runtime.ScalaRunTime.stringOf
    // import scalation.scala2d.writeImage

    // // val xyr_fname = Array ("displacement", "cylinders", "horsepower", "weight", "acceleration", "modelyear", "origin", "mpg")
    // // val xr_fname = Array ("displacement", "cylinders", "horsepower", "weight", "acceleration", "modelyear", "origin")
    // // val x_fname = Array ("displacement", "cylinders", "horsepower", "weight", "acceleration", "modelyear")
    // val ox_fname = Array ("intercept", "displacement", "cylinders", "horsepower", "weight", "acceleration", "modelyear")
    // // val y_fname = Array ("mpg")
    // // val xy_fname = Array ("displacement", "cylinders", "horsepower", "weight", "acceleration", "modelyear", "mpg")
    // // val oxy_fname = Array ("intercept", "displacement", "cylinders", "horsepower", "weight", "acceleration", "modelyear", "mpg")

    // val xyr = MatrixD.load ("auto_mpg_cleaned.csv", 1, sp=',')      // load the dataset, skipping the header row
    // val xr = xyr(?, 0 until 6)                                  // get the first 6 columns as the feature matrix
    // val x = xr(?, 0 until 5)                                  // get the first 5 columns as the feature matrix
    // val y = xyr(?, 7)                                        // get the 7th column as the response vector
    // val _1     = VectorD.one (xyr.dim)                                  // vector of all ones    
    // val ox     = _1 +^: x                                              // prepend a column of all ones to x
    // val yy     = MatrixD.fromVector (y)                                // turn the m-vector y into an m-by-1 matrix 
    // val xy = xyr.not(?, 6)                                             // remove the origin column
    // val oxy = _1 +^: xy                                              // prepend a column of all ones to the first 7 columns of xy



    val n_q = 15
    val r_q = 0 until 15
    val colName = "Regression, In-Sample, 80-20 Split"
    val rowName = modeling.qoF_names.take(n_q)  



    banner ("In-Sample")                 
    var reg = new Regression (ox, y, ox_fname)                               // Regression model
    val qof1 = reg.trainNtest ()()._2(r_q)                     // train and test on full dataset
    println (reg.summary ())                                       // parameter/coefficient statistics
    


    banner("80-20 Split")
    val permGen = scalation.mathstat.TnT_Split.makePermGen (oxy.dim)             // make a permutation generator
    val n_test = (oxy.dim * 0.2).toInt                                           // 80% training, 20% testing
    val idx = scalation.mathstat.TnT_Split.testIndices(permGen, n_test)         // get test indices for 80-20 split
    val (ox_test, ox_train) = TnT_Split (ox, idx)                               // TnT split the dataset ox (row split)
    val (yy_test, yy_train) = TnT_Split (yy, idx)                               // TnT split the response vector y (row split)
    val y_train = yy_train.col(0)                                                     // get the test response vector from the test response matrix
    val y_test = yy_test.col(0)                                                       // get the test response vector from the test response matrix
    reg = new Regression (ox, y, ox_fname)                       // Regression model
    val qof2 = reg.trainNtest (ox_train,y_train)(ox_test, y_test)._2(r_q)              // train on random 80%, test on whole dataset
    println (reg.summary ())                                       // parameter/coefficient statistics


    reg = new Regression (ox, y, ox_fname)                       // Regression model

    banner("5-Fold CV")
    banner ("Cross-Validation")
    FitM.showQofStatTable (reg.crossValidate ())

    val caption = "AutoMPG Linear Regression"
    val name    = "Linear Regression"
    val qofs    = MatrixD (qof1, qof2).transpose            // create metrics for both point and interval predictions
     
    val latex   = make_doc (make_table (caption, name, qofs, colName, rowName))
    println (latex)

end LinRegAutoMPG



@main def LinReghouse (): Unit =

    import scalation.mathstat._
    import scalation.mathstat.Scala2LaTeX._
    // import scala.runtime.ScalaRunTime.stringOf

    // val xy_fname = Array ("Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality", "House_Price")
    // val x_fname = Array ("Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality")
    val ox_fname = Array ("intercept", "Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality")
    // val y_fname = Array ("House_Price")
    // val oxy_fname = Array ("intercept", "Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality", "House_Price")


    val xy = MatrixD.load ("house_price_regression_dataset.csv", 1, sp=',')      // load the dataset, skipping the header row
    val x = xy(?, 0 until 6)                                  // get the first 6 columns as the feature matrix
    val y = xy(?, 7)                                        // get the 7th column as the response vector
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones    
    val ox     = _1 +^: x                                              // prepend a column of all ones to x
    val yy     = MatrixD.fromVector (y)                                // turn the m-vector y into an m-by-1 matrix 
    val oxy = _1 +^: xy                                              // prepend a column of all ones to the first 7 columns of xy



    val n_q = 15
    val r_q = 0 until 15
    val colName = "Regression, In-Sample, 80-20 Split"
    val rowName = modeling.qoF_names.take(n_q)  



    banner ("In-Sample")                 
    var reg = new Regression (ox, y, ox_fname)                               // Regression model
    val qof1 = reg.trainNtest ()()._2(r_q)                     // train and test on full dataset
    println (reg.summary ())                                       // parameter/coefficient statistics



    banner("80-20 Split")
    val permGen = scalation.mathstat.TnT_Split.makePermGen (oxy.dim)             // make a permutation generator
    val n_test = (oxy.dim * 0.2).toInt                                           // 80% training, 20% testing
    val idx = scalation.mathstat.TnT_Split.testIndices(permGen, n_test)         // get test indices for 80-20 split
    val (ox_test, ox_train) = TnT_Split (ox, idx)                               // TnT split the dataset ox (row split)
    val (yy_test, yy_train) = TnT_Split (yy, idx)                               // TnT split the response vector y (row split)
    val y_train = yy_train.col(0)                                                     // get the test response vector from the test response matrix
    val y_test = yy_test.col(0)                                                       // get the test response vector from the test response matrix
    reg = new Regression (ox, y, ox_fname)                       // Regression model
    val qof2 = reg.trainNtest (ox_train,y_train)(ox_test, y_test)._2(r_q)              // train on random 80%, test on 20%
    println (reg.summary ())                                       // parameter/coefficient statistics


    reg = new Regression (ox, y, ox_fname)                       // Regression model

    banner("5-Fold CV")
    banner ("Cross-Validation")
    FitM.showQofStatTable (reg.crossValidate ())

    val caption = "House Price Linear Regression"
    val name    = "Linear Regression"
    val qofs    = MatrixD (qof1, qof2).transpose            // create metrics for both point and interval predictions
     
    val latex   = make_doc (make_table (caption, name, qofs, colName, rowName))
    println (latex)

end LinReghouse



@main def LinReginsurance (): Unit =

    import scalation.mathstat._
    import scalation.mathstat.Scala2LaTeX._
    // import scala.runtime.ScalaRunTime.stringOf

    // val oxy_fname = Array ("intercept", "age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest", "charges")
    val ox_fname = Array ("intercept", "age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest")
    // val y_fname = Array ("charges")


    val oxy = MatrixD.load ("insurance_cat2num.csv", 1, sp=',')      // load the dataset, skipping the header row
    val ox = oxy(?, 0 until 8)                                  // get the first 8 columns as the feature matrix
    val y = oxy(?, 9)                                        // get the 9th column as the response vector
    val yy     = MatrixD.fromVector (y)                                // turn the m-vector y into an m-by-1 matrix 



    val n_q = 15
    val r_q = 0 until 15
    val colName = "Regression, In-Sample, 80-20 Split"
    val rowName = modeling.qoF_names.take(n_q)  



    banner ("In-Sample")                 
    var reg = new Regression (ox, y, ox_fname)                               // Regression model
    val qof1 = reg.trainNtest ()()._2(r_q)                     // train and test on full dataset
    println (reg.summary ())                                       // parameter/coefficient statistics



    banner("80-20 Split")
    val permGen = scalation.mathstat.TnT_Split.makePermGen (oxy.dim)             // make a permutation generator
    val n_test = (oxy.dim * 0.2).toInt                                           // 80% training, 20% testing
    val idx = scalation.mathstat.TnT_Split.testIndices(permGen, n_test)         // get test indices for 80-20 split
    val (ox_test, ox_train) = TnT_Split (ox, idx)                               // TnT split the dataset ox (row split)
    val (yy_test, yy_train) = TnT_Split (yy, idx)                               // TnT split the response vector y (row split)
    val y_train = yy_train.col(0)                                                     // get the test response vector from the test response matrix
    val y_test = yy_test.col(0)                                                       // get the test response vector from the test response matrix
    reg = new Regression (ox, y, ox_fname)                       // Regression model
    val qof2 = reg.trainNtest (ox_train,y_train)(ox_test, y_test)._2(r_q)              // train on random 80%, test on 20%
    println (reg.summary ())                                       // parameter/coefficient statistics


    reg = new Regression (ox, y, ox_fname)                       // Regression model

    banner("5-Fold CV")
    banner ("Cross-Validation")
    FitM.showQofStatTable (reg.crossValidate ())

    val caption = "Insurance Charges Linear Regression"
    val name    = "Linear Regression"
    val qofs    = MatrixD (qof1, qof2).transpose            // create metrics for both point and interval predictions
     
    val latex   = make_doc (make_table (caption, name, qofs, colName, rowName))
    println (latex)

end LinReginsurance