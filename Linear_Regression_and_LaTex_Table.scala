package scalation
package modeling

@main def LinRegAutoMPG (): Unit =

    import Example_AutoMPG._
    import scalation.mathstat._
    import scalation.mathstat.Scala2LaTeX._
    import scala.runtime.ScalaRunTime.stringOf

    val n_q = 15
    val r_q = 0 until 15
    val colName = "Regression, In-Sample, 80-20 Split"
    val rowName = modeling.qoF_names.take(n_q)  



    banner ("In-Sample")                 
    var reg = new Regression (ox, y)                               // Regression model
    val qof1 = reg.trainNtest ()()._2(r_q)                         // train and test on full dataset
    println (reg.summary ())                                       // parameter/coefficient statistics



    banner("80-20 Split")
    val permGen = scalation.mathstat.TnT_Split.makePermGen (oxy.dim)             // make a permutation generator
    val n_test = (oxy.dim * 0.2).toInt                                           // 80% training, 20% testing
    val idx = scalation.mathstat.TnT_Split.testIndices(permGen, n_test)         // get test indices for 80-20 split
    val (_, ox_train) = TnT_Split (ox, idx)                               // TnT split the dataset ox (row split)
    val (_, yy_train) = TnT_Split (yy, idx)                               // TnT split the response vector y (row split)
    val y_train = yy_train.col(0)                                                     // get the test response vector from the test response matrix
    reg = new Regression (ox, y)                       // Regression model, train on 80% training data
    val qof2 = reg.trainNtest (ox_train,y_train)()._2(r_q)                       // train on random 80%, test on whole dataset to compare with in-sample results
    println (reg.summary ())                                       // parameter/coefficient statistics



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