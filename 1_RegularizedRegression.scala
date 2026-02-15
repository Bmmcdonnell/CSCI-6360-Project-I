package scalation.modeling

import scalation.mathstat._
import scalation.mathstat.Scala2LaTeX._

@main def regRegressionAutoMPG (): Unit = 
    import Example_AutoMPG._

    val n_q = 15
    val r_q = 0 until 15
    val rowName = qoF_names.take(n_q)

    val ridreg = RidgeRegression.center(x, y, x_fname)
    val qof1 = ridreg.trainNtest ()()._2(r_q)                                              // train and test the model
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
    
    val lassoReg = new LassoRegression (x, y, x_fname)                    // create a Lasso regression model
    val qof2 = lassoReg.trainNtest ()()._2(r_q)                                            // train and test the model
    println (lassoReg.summary ())                                       // parameter/coefficient statistics

    //banner ("Forward Selection Test")
    val (cols, rSq) = lassoReg.forwardSelAll ()                         // R^2, R^2 bar, sMAPE, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.ᵀ, Regression.metrics, "R^2 vs n for LassoRegression", lines = true)
    println (s"rSq = $rSq")
    
    val qofsAuto = MatrixD (qof1, qof2).transpose

    //Latex
    val latex = make_doc (make_table ("Lasso and Ridge Regression AutoMPG results", "Regularized Regression on AutoMPG", qofsAuto, "Ridge, Lasso", rowName))
    println(latex)
end regRegressionAutoMPG

@main def regRegressionHousePrice (): Unit = 
   import scalation.database.table.Table

    val n_q = 15
    val r_q = 0 until 15
    val rowName = qoF_names.take(n_q)

    val data = Table.load("house_price_regression_dataset.csv", "house_price_regression_dataset", 8, null)
    val xcols  = Array (1, 2, 3, 4, 5, 6, 7)
    val (x, y) = data.toMatrixV (xcols, 2)
    val fname  = xcols.map (data.schema (_))
    println (s"fname = $fname")
    println (s"y = $y")
    println (s"y = $y")

    //Ridge Regression

    val ridreg = RidgeRegression.center(x, y, fname)
    val qof3 = ridreg.trainNtest ()()._2(r_q)                                                // train and test the model
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

    //Lasso Regression
    val lassoReg = new LassoRegression (x, y, fname)                    // create a Lasso regression model
    val qof4 = lassoReg.trainNtest ()()._2(r_q)                                            // train and test the model
    println (lassoReg.summary ())                                       // parameter/coefficient statistics

    //banner ("Forward Selection Test")
    val (cols, rSq) = lassoReg.forwardSelAll ()                         // R^2, R^2 bar, sMAPE, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.ᵀ, Regression.metrics, "R^2 vs n for LassoRegression", lines = true)
    println (s"rSq = $rSq")

    val qofsHouse = MatrixD (qof3, qof4).transpose
    //Latex
    val latex = make_doc (make_table ("Lasso and Ridge Regression House Prices results", "Regularized Regression on House Prices", qofsHouse, "Ridge, Lasso", rowName))
    println(latex)

end regRegressionHousePrice

@main def regRegressionInsurance (): Unit = 
    print("TBD")
end regRegressionInsurance