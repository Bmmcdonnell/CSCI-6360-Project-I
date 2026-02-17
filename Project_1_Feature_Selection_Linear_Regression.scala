package scalation
package modeling

import scalation.mathstat._



def LinRegFS (ox: MatrixD,  y: VectorD, ox_fname: Array[String]): Unit =

    // 1. Run Forward Selection to get the order of important columns
    val reg = new Regression (ox, y, ox_fname)
    val (cols, rSq) = reg.forwardSelAll () // cols = indices of selected features

    // 2. Calculate AIC for each stage of the selection
    //    We loop from 1 feature up to all features selected
    val aic_list = new VectorD (cols.size)


    for k <- 0 until cols.size do
        // Create a subset of the matrix with the top k+1 features
        val subCols = cols.slice (0, k + 1)
        val ox_sub = ox(?, subCols)

        // Fit a new model on this subset
        val subReg = new Regression (ox_sub, y)
        val (_, qof) = subReg.trainNtest ()()
        
        // Store the AIC for this model
        aic_list(k) = qof(13)
    end for

    banner("AIC History")
    val colsArray = cols.toArray
    val colsArrayD = colsArray.map (_.toDouble)
    val colsVecD = VectorD (colsArrayD) // convert Seq[Int] to VectorD for better display
    val aic_matrix    = MatrixD (colsVecD, aic_list)            // convert AIC vector to a 2-row matrix with the top row being which variable was added at that step for better display
    println (s"AIC History:      $aic_matrix")

    // 3. Plotting

    // Plot 1: R^2 vs n
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Linear Rgression Forward Selection", lines = true)

    // Plot 2: AIC vs n
    new PlotM (null, MatrixD.fromVector (aic_list).ᵀ, Array ("AIC"), "AIC vs n for Linear Regression Forward Selection", lines = true)

end LinRegFS


def LinRegBE (ox: MatrixD,  y: VectorD, ox_fname: Array[String]): Unit =

    // 1. Run Forward Selection to get the order of important columns
    val reg = new Regression (ox, y, ox_fname)
    val (cols, rSq) = reg.backwardElimAll () // cols = indices of selected features

    // 2. Calculate AIC for each stage of the selection
    //    We loop from 1 feature up to all features selected
    val aic_list = new VectorD (cols.size)


    for k <- 0 until cols.size do
        // Create a subset of the matrix with the top k+1 features
        val subCols = cols.slice (0, k + 1)
        val ox_sub = ox(?, subCols)

        // Fit a new model on this subset
        val subReg = new Regression (ox_sub, y)
        val (_, qof) = subReg.trainNtest ()()
        
        // Store the AIC for this model
        aic_list(k) = qof(13)
    end for

    banner("AIC History")
    val colsArray = cols.toArray
    val colsArrayD = colsArray.map (_.toDouble)
    val colsVecD = VectorD (colsArrayD) // convert Seq[Int] to VectorD for better display
    val aic_matrix    = MatrixD (colsVecD, aic_list)            // convert AIC vector to a 2-row matrix with the top row being which variable was added at that step for better display
    println (s"AIC History:      $aic_matrix")

    // 3. Plotting

    // Plot 1: R^2 vs n
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Linear Rgression Backward Elimination", lines = true)

    // Plot 2: AIC vs n
    new PlotM (null, MatrixD.fromVector (aic_list).ᵀ, Array ("AIC"), "AIC vs n for Linear Regression Backward Elimination", lines = true)

end LinRegBE



def LinRegSS (ox: MatrixD,  y: VectorD, ox_fname: Array[String]): Unit =

    // 1. Run Forward Selection to get the order of important columns
    val reg = new Regression (ox, y, ox_fname)
    val (cols, rSq) = reg.stepwiseSelAll () // cols = indices of selected features

    // 2. Calculate AIC for each stage of the selection
    //    We loop from 1 feature up to all features selected
    val aic_list = new VectorD (cols.size)


    for k <- 0 until cols.size do
        // Create a subset of the matrix with the top k+1 features
        val subCols = cols.slice (0, k + 1)
        val ox_sub = ox(?, subCols)

        // Fit a new model on this subset
        val subReg = new Regression (ox_sub, y)
        val (_, qof) = subReg.trainNtest ()()
        
        // Store the AIC for this model
        aic_list(k) = qof(13)
    end for

    banner("AIC History")
    val colsArray = cols.toArray
    val colsArrayD = colsArray.map (_.toDouble)
    val colsVecD = VectorD (colsArrayD) // convert Seq[Int] to VectorD for better display
    val aic_matrix    = MatrixD (colsVecD, aic_list)            // convert AIC vector to a 2-row matrix with the top row being which variable was added at that step for better display
    println (s"AIC History:      $aic_matrix")

    // 3. Plotting

    // Plot 1: R^2 vs n
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Linear Rgression Stepwise Selection", lines = true)

    // Plot 2: AIC vs n
    new PlotM (null, MatrixD.fromVector (aic_list).ᵀ, Array ("AIC"), "AIC vs n for Linear Regression Stepwise Selection", lines = true)

end LinRegSS


@main def LinRegAutoMPGFS (): Unit =
    import Example_AutoMPG._
    LinRegFS(ox, y, ox_fname)
end LinRegAutoMPGFS

@main def LinRegAutoMPGBE (): Unit =
    import Example_AutoMPG._
    LinRegBE(ox, y, ox_fname)
end LinRegAutoMPGBE

@main def LinRegAutoMPGSS (): Unit =
    import Example_AutoMPG._
    LinRegSS(ox, y, ox_fname)
end LinRegAutoMPGSS

@main def LinReghouseFS (): Unit =
    val ox_fname = Array ("intercept", "Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality")
    val xy = MatrixD.load ("house_price_regression_dataset.csv", 1, sp=',')      // load the dataset, skipping the header row
    val x = xy(?, 0 until 6)                                  // get the first 6 columns as the feature matrix
    val y = xy(?, 7)                                        // get the 7th column as the response vector
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones    
    val ox     = _1 +^: x                                              // prepend a column of all ones to x
    LinRegFS(ox, y, ox_fname)
end LinReghouseFS

@main def LinReghouseBE (): Unit =
    val ox_fname = Array ("intercept", "Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality")
    val xy = MatrixD.load ("house_price_regression_dataset.csv", 1, sp=',')      // load the dataset, skipping the header row
    val x = xy(?, 0 until 6)                                  // get the first 6 columns as the feature matrix
    val y = xy(?, 7)                                        // get the 7th column as the response vector
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones    
    val ox     = _1 +^: x                                              // prepend a column of all ones to x
    LinRegBE(ox, y, ox_fname)
end LinReghouseBE

@main def LinReghouseSS (): Unit =
    val ox_fname = Array ("intercept", "Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality")
    val xy = MatrixD.load ("house_price_regression_dataset.csv", 1, sp=',')      // load the dataset, skipping the header row
    val x = xy(?, 0 until 6)                                  // get the first 6 columns as the feature matrix
    val y = xy(?, 7)                                        // get the 7th column as the response vector
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones    
    val ox     = _1 +^: x                                              // prepend a column of all ones to x
    LinRegSS(ox, y, ox_fname)
end LinReghouseSS

@main def LinReginsuranceFS (): Unit =
    val ox_fname = Array ("intercept", "age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest")
    val oxy = MatrixD.load ("insurance_cat2num.csv", 1, sp=',')      // load the dataset, skipping the header row
    val ox = oxy(?, 0 until 8)                                  // get the first 8 columns as the feature matrix
    val y = oxy(?, 9)                                        // get the 9th column as the response vector
    LinRegFS(ox, y, ox_fname)
end LinReginsuranceFS

@main def LinReginsuranceBE (): Unit =
    val ox_fname = Array ("intercept", "age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest")
    val oxy = MatrixD.load ("insurance_cat2num.csv", 1, sp=',')      // load the dataset, skipping the header row
    val ox = oxy(?, 0 until 8)                                  // get the first 8 columns as the feature matrix
    val y = oxy(?, 9)                                        // get the 9th column as the response vector
    LinRegBE(ox, y, ox_fname)
end LinReginsuranceBE

@main def LinReginsuranceSS (): Unit =
    val ox_fname = Array ("intercept", "age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest")
    val oxy = MatrixD.load ("insurance_cat2num.csv", 1, sp=',')      // load the dataset, skipping the header row
    val ox = oxy(?, 0 until 8)                                  // get the first 8 columns as the feature matrix
    val y = oxy(?, 9)                                        // get the 9th column as the response vector
    LinRegSS(ox, y, ox_fname)
end LinReginsuranceSS