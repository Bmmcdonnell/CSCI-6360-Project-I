package scalation
package modeling

@main def Project_1_Linear_Regression (): Unit =

    import Example_AutoMPG._
    import scalation.mathstat._
    import scalation.mathstat.Scala2LaTeX._

    // ox - X matrix
    // y  - response vector
    // ox_fname - feature/column names for X matrix

    val n_q = 15
    val r_q = 0 until 15

    banner ("Regression Model")
    val mod1 = modeling.Regression (xy)()                             // Regression model
    val qof1 = mod1.trainNtest ()()._2(r_q)                           // train and test on full dataset
    println (mod1.summary ())                        // parameter/coefficient statistics


    banner ("Ridge Regression Model")
    val mod2 =  RidgeRegression (xy)()                       // Ridge Regression model
    val qof_ = mod2.trainNtest ()()._2(r_q)                           // train and test on full dataset
    val qof2 = modeling.RidgeRegression.fix_smape (mod2, y, qof_)
    println (mod2.summary ())                                // parameter/coefficient statistics

    val caption = "AutoMPG Linear Regression"
    val name    = "Linear Regression"
    val qofs    = MatrixD (qof1, qof2).transpose            // create metrics for both point and interval predictions
    // val colName = "Linear Regression"
    val colName = "Metric, Regression"
    // val rowName = Array ("MAPE", "SMAPE", "R^2", "RMSE", "MAE")
    
    val rowName = modeling.qoF_names.take(n_q)  
    val latex   = make_doc (make_table (caption, name, qofs, colName, rowName))
    println (latex)

end Project_1_Linear_Regression