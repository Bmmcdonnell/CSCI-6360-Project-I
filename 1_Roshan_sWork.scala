package scalation.modeling

import scalation.mathstat._

@main def ridgeRegressionAutoMPG (): Unit = 
    import Example_AutoMPG._

    val ridreg = RidgeRegression.center(x, y, x_fname)
    ridreg.trainNtest ()()                                                // train and test the model
    println (ridreg.summary ())

    FitM.showQofStatTable (ridreg.crossValidate ())

    println (s"x_fname = ${(x_fname)}")

    for tech <- SelectionTech.values do
        //banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = ridreg.selectFeatures (tech)                    // R^2, R^2 bar, R^2 cv
        val k = cols.size
        val t = VectorD.range (1, k)     
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (t, rSq.ᵀ, Regression.metrics, s"R^2 vs n for RidgeRegression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for
end ridgeRegressionAutoMPG

@main def lassoRegressionAutoMPG (): Unit = 
    import Example_AutoMPG._

    val lassoReg = new LassoRegression (x, y, x_fname)                    // create a Lasso regression model
    lassoReg.trainNtest ()()                                            // train and test the model
    println (lassoReg.summary ())                                       // parameter/coefficient statistics

    //banner ("Forward Selection Test")
    val (cols, rSq) = lassoReg.forwardSelAll ()                         // R^2, R^2 bar, sMAPE, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.ᵀ, Regression.metrics, "R^2 vs n for LassoRegression", lines = true)
    println (s"rSq = $rSq")
end lassoRegressionAutoMPG

@main def ridgeRegressionHousePrice (): Unit = 
   import scalation.database.table.Table

    val data = Table.load("house_price_regression_dataset.csv", "house_price_regression_dataset", 8, null)
    val xcols  = Array (1, 2, 3, 4, 5, 6, 7)
    val (x, y) = data.toMatrixV (xcols, 2)
    val fname  = xcols.map (data.schema (_))
    println (s"fname = $fname")
    println (s"y = $y")
    println (s"y = $y")

    val ridreg = RidgeRegression.center(x, y, fname)
    ridreg.trainNtest ()()                                                // train and test the model
    println (ridreg.summary ())

    FitM.showQofStatTable (ridreg.crossValidate ())

    println (s"fname = ${(fname)}")

    for tech <- SelectionTech.values do
        //banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = ridreg.selectFeatures (tech)                    // R^2, R^2 bar, R^2 cv
        val k = cols.size
        val t = VectorD.range (1, k)     
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (t, rSq.ᵀ, Regression.metrics, s"R^2 vs n for RidgeRegression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for
end ridgeRegressionHousePrice

@main def lassoRegressionHousePrice (): Unit =
    import scalation.database.table.Table
    val data = Table.load("house_price_regression_dataset.csv", "house_price_regression_dataset", 8, null)
    val xcols  = Array (1, 2, 3, 4, 5, 6, 7)
    val (x, y) = data.toMatrixV (xcols, 2)
    val fname  = xcols.map (data.schema (_))
    println (s"fname = $fname")
    println (s"y = $y")
    println (s"y = $y")

    val lassoReg = new LassoRegression (x, y, fname)                    // create a Lasso regression model
    lassoReg.trainNtest ()()                                            // train and test the model
    println (lassoReg.summary ())                                       // parameter/coefficient statistics

    //banner ("Forward Selection Test")
    val (cols, rSq) = lassoReg.forwardSelAll ()                         // R^2, R^2 bar, sMAPE, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.ᵀ, Regression.metrics, "R^2 vs n for LassoRegression", lines = true)
    println (s"rSq = $rSq")
end lassoRegressionHousePrice