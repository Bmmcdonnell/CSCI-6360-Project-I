package scalation
package modeling

@main def LinRegAutoMPG (): Unit =

    import Example_AutoMPG._
    import scalation.mathstat._
    import scalation.mathstat.Scala2LaTeX._
    import scala.runtime.ScalaRunTime.stringOf

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
    val (yp1, qof1_full) = reg.trainNtest ()()                     // train and test on full dataset
    val qof1 = qof1_full(r_q)                                      // slice QoF to first 15 metrics
    println (reg.summary ())                                       // parameter/coefficient statistics
    Predictor.plotPrediction(y, yp1, reg.modelName)               // plot predicted vs actual values for in-sample predictions



    banner("80-20 Split")
    val permGen = scalation.mathstat.TnT_Split.makePermGen (oxy.dim)             // make a permutation generator
    val n_test = (oxy.dim * 0.2).toInt                                           // 80% training, 20% testing
    val idx = scalation.mathstat.TnT_Split.testIndices(permGen, n_test)         // get test indices for 80-20 split
    val (_, ox_train) = TnT_Split (ox, idx)                               // TnT split the dataset ox (row split)
    val (_, yy_train) = TnT_Split (yy, idx)                               // TnT split the response vector y (row split)
    val y_train = yy_train.col(0)                                                     // get the test response vector from the test response matrix
    reg = new Regression (ox, y, ox_fname)                       // Regression model
    val (yp2, qof2_full) = reg.trainNtest (ox_train,y_train)()              // train on random 80%, test on whole dataset
    val qof2 = qof2_full(r_q)                                      // slice QoF to first 15 metrics
    println (reg.summary ())                                       // parameter/coefficient statistics
    Predictor.plotPrediction(y, yp2, reg.modelName)            // plot predicted vs actual values for 80-20 split predictions  


    reg = new Regression (ox, y, ox_fname)                       // Regression model

    banner("5-Fold CV")
    banner ("Cross-Validation")
    FitM.showQofStatTable (reg.crossValidate ())

    println (s"ox_fname = ${stringOf (ox_fname)}")

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = reg.selectFeatures (tech)           // R^2, R^2 bar, sMAPE, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Regression with $tech", lines = true)
        banner ("Feature Importance")
        println (s"$tech: rSq = $rSq")
        val imp = reg.importance (cols.toArray, rSq)
        for (c, r) <- imp do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r") 
    end for


    val caption = "AutoMPG Linear Regression"
    val name    = "Linear Regression"
    val qofs    = MatrixD (qof1, qof2).transpose            // create metrics for both point and interval predictions
     
    val latex   = make_doc (make_table (caption, name, qofs, colName, rowName))
    println (latex)

end LinRegAutoMPG



@main def LinReghouse (): Unit =

    import scalation.mathstat._
    import scalation.mathstat.Scala2LaTeX._
    import scala.runtime.ScalaRunTime.stringOf

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
    val (yp1, qof1_full) = reg.trainNtest ()()                     // train and test on full dataset
    val qof1 = qof1_full(r_q)                                      // slice QoF to first 15 metrics
    println (reg.summary ())                                       // parameter/coefficient statistics
    Predictor.plotPrediction(y, yp1, reg.modelName)               // plot predicted vs actual values for in-sample predictions



    banner("80-20 Split")
    val permGen = scalation.mathstat.TnT_Split.makePermGen (oxy.dim)             // make a permutation generator
    val n_test = (oxy.dim * 0.2).toInt                                           // 80% training, 20% testing
    val idx = scalation.mathstat.TnT_Split.testIndices(permGen, n_test)         // get test indices for 80-20 split
    val (_, ox_train) = TnT_Split (ox, idx)                               // TnT split the dataset ox (row split)
    val (_, yy_train) = TnT_Split (yy, idx)                               // TnT split the response vector y (row split)
    val y_train = yy_train.col(0)                                                     // get the test response vector from the test response matrix
    reg = new Regression (ox, y, ox_fname)                       // Regression model
    val (yp2, qof2_full) = reg.trainNtest (ox_train,y_train)()              // train on random 80%, test on whole dataset
    val qof2 = qof2_full(r_q)                                      // slice QoF to first 15 metrics
    println (reg.summary ())                                       // parameter/coefficient statistics
    Predictor.plotPrediction(y, yp2, reg.modelName)            // plot predicted vs actual values for 80-20 split predictions  


    reg = new Regression (ox, y, ox_fname)                       // Regression model

    banner("5-Fold CV")
    banner ("Cross-Validation")
    FitM.showQofStatTable (reg.crossValidate ())

    println (s"ox_fname = ${stringOf (ox_fname)}")

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = reg.selectFeatures (tech)           // R^2, R^2 bar, sMAPE, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Regression with $tech", lines = true)
        banner ("Feature Importance")
        println (s"$tech: rSq = $rSq")
        val imp = reg.importance (cols.toArray, rSq)
        for (c, r) <- imp do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r") 
    end for


    val caption = "House Price Linear Regression"
    val name    = "Linear Regression"
    val qofs    = MatrixD (qof1, qof2).transpose            // create metrics for both point and interval predictions
     
    val latex   = make_doc (make_table (caption, name, qofs, colName, rowName))
    println (latex)

end LinReghouse