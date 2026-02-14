// Paste this at the end of Scala2LaTeX.scala

// Reduce the page margin by adding the following line in the LaTeX file
// \usepackage[a4paper, margin=0.5in]{geometry}


@main def autoMPGTranRegression(): Unit =
    import Scala2LaTeX._
    import scala.math.{sqrt, log1p, expm1}
    import modeling.TranRegression.{box_cox, cox_box}
    import modeling.Example_AutoMPG._
    val n_q = 15                                                      // number of core metrics
    val r_q = 0 until 15                                              // range of core metrics
    //val colName = "Metric, sqrt, log1p, box-cox(λ=0.2), box-cox(λ=0.3), box-cox(λ=0.4), box-cox(λ=0.5), box-cox(λ=0.6)"  // column names -- which models
    val colName = "Metric, sqrt, log1p, box-cox($\\lambda$=0.2), box-cox($\\lambda$=0.3), box-cox($\\lambda$=0.4), box-cox($\\lambda$=0.5), box-cox($\\lambda$=0.6)"
    val rowName = modeling.qoF_names.take(n_q)      
   
    val xform = List((sqrt, sq, "sqrt"), (log1p, expm1, "log1p"), (box_cox, cox_box, "box_cox"))
    val qofB = scala.collection.mutable.ArrayBuffer[scalation.mathstat.VectorD]()
    //val qofs = new MatrixD(n_q, 8)

    for f <- xform do
        if f._3 != "box_cox" then
            banner (s"TranRegression with ${f._3} transform")
            val mod = new modeling.TranRegression (ox, y, ox_fname, modeling.Regression.hp, f._1, f._2)                   
            val qof = mod.trainNtest ()()._2(r_q)                           // train and test on full dataset
            qofB += qof
            println (mod.summary ())
        else 
            for lamda <- List(0.2, 0.3, 0.4, 0.5, 0.6) do
                modeling.TranRegression.λ = lamda   
                banner (s"TranRegression with ${f._3}(λ = ${lamda}) transform")
                val mod = new modeling.TranRegression (ox, y, ox_fname, modeling.Regression.hp, f._1, f._2)
                val qof = mod.trainNtest ()()._2(r_q)                           // train and test on full dataset
                qofB += qof
                println (mod.summary ())

    banner ("Copy-paste the LateX table below into a .tex file")
    //val qofs = MatrixD (qof1, qof2, qof3, qof4, qof5).transpose       // put QoFs in a matrix
    val caption = "Auto MPG Regression: Sqrt, Log1p, and Box–Cox($\\lambda$ = 0.2 to 0.6)"   // LaTex figure caption
    val name    = "TranRegression"
    val qofs = MatrixD(qofB.toSeq*).transpose
    val latex   = make_doc (make_table (caption, name, qofs, colName, rowName))
    println (latex)

end autoMPGTranRegression


@main def housePriceTranRegression(): Unit =
    import Scala2LaTeX._
    import scala.math.{sqrt, log1p, expm1}
    import modeling.TranRegression.{box_cox, cox_box}
    
    val n_q = 15                                                      // number of core metrics
    val r_q = 0 until 15                                              // range of core metrics
    //val colName = "Metric, sqrt, log1p, box-cox(λ=0.2), box-cox(λ=0.3), box-cox(λ=0.4), box-cox(λ=0.5), box-cox(λ=0.6)"  // column names -- which models
    val colName = "Metric, sqrt, log1p, box-cox($\\lambda$=0.2), box-cox($\\lambda$=0.3), box-cox($\\lambda$=0.4), box-cox($\\lambda$=0.5), box-cox($\\lambda$=0.6)"
    val rowName = modeling.qoF_names.take(n_q)      
   
    val xform = List((sqrt, sq, "sqrt"), (log1p, expm1, "log1p"), (box_cox, cox_box, "box_cox"))
    val qofB = scala.collection.mutable.ArrayBuffer[scalation.mathstat.VectorD]()
    //val qofs = new MatrixD(n_q, 8)

    val xy = MatrixD.load ("boston_with_intercept.csv", 1, 0)
    val ox = xy(?, 0 until 8)
    val y = xy(?, 8)
    val ox_fname = Array ("_1", "x1", "x2", "x3", "x4", "x5", "x6", "x7")   

    for f <- xform do
        if f._3 != "box_cox" then
            banner (s"TranRegression with ${f._3} transform")
            val mod = new modeling.TranRegression (ox, y, ox_fname, modeling.Regression.hp, f._1, f._2)                   
            val qof = mod.trainNtest ()()._2(r_q)                           // train and test on full dataset
            qofB += qof
            println (mod.summary ())
        else 
            for lamda <- List(0.2, 0.3, 0.4, 0.5, 0.6) do
                modeling.TranRegression.λ = lamda   
                banner (s"TranRegression with ${f._3}(λ = ${lamda}) transform")
                val mod = new modeling.TranRegression (ox, y, ox_fname, modeling.Regression.hp, f._1, f._2)
                val qof = mod.trainNtest ()()._2(r_q)                           // train and test on full dataset
                qofB += qof
                println (mod.summary ())

    banner ("Copy-paste the LateX table below into a .tex file")
    //val qofs = MatrixD (qof1, qof2, qof3, qof4, qof5).transpose       // put QoFs in a matrix
    val caption = "House Price Regression: Sqrt, Log1p, and Box–Cox($\\lambda$ = 0.2 to 0.6)"   // LaTex figure caption
    val name    = "TranRegression"
    val qofs = MatrixD(qofB.toSeq*).transpose
    val latex   = make_doc (make_table (caption, name, qofs, colName, rowName))
    println (latex)

end housePriceTranRegression

