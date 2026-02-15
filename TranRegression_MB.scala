package scalation
package modeling

import scalation.mathstat._ 
import scala.math.{sqrt, log1p, expm1}
import modeling.TranRegression.{box_cox, cox_box}
import scalation.mathstat.Scala2LaTeX._
import scala.collection.mutable.LinkedHashSet

def display(key: String, res: (LinkedHashSet[Int], MatrixD), f_name: String, n_features: Int): Unit =
    val (cols, rSq) = res
    val k = cols.size
    println (s"k = $k, n = $n_features")
    println (s"rSq = $rSq")
    if key != "stepwise" then
        new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for TranRegression $f_name", lines = true)

def model(mod: modeling.TranRegression, key: String, qofB: scala.collection.mutable.ArrayBuffer[scalation.mathstat.VectorD], f_name: String, n_features: Int): Unit =
    val r_q = 0 until 15
    if key == "train" then
        banner ("Training using the full data")
        qofB += mod.trainNtest ()()._2(r_q)   
        //println (mod.summary ())
    else if key == "val" then
        banner ("Validation (80-20 train-test-split)")
        qofB += mod.validate ()()._2(r_q) 
    else if key == "forward" then
        banner (s"Forward Selection Test")
        display(key, mod.forwardSelAll (), f_name, n_features)
    else if key == "backward" then
        banner ("Backward Selection Test")
        display(key, mod.backwardElimAll (), f_name, n_features)
    else if key == "stepwise" then
        banner ("Stepwise Selection Test")
        display(key, mod.stepwiseSelAll (), f_name, n_features)
    else
        mod.inSample_Test()

def myTranRegression(ox: MatrixD, y: VectorD, ox_fname: Array[String], key: String, title: String): Unit =
    val n_q = 15                                                      // number of core metrics
    //val r_q = 0 until 15                                              // range of core metrics
    val colName = "Metric, sqrt, log1p, box-cox($\\lambda$=0.2), box-cox($\\lambda$=0.3), box-cox($\\lambda$=0.4), box-cox($\\lambda$=0.5), box-cox($\\lambda$=0.6)"
    val rowName = modeling.qoF_names.take(n_q)      
   
    val xform = List((sqrt, sq, "sqrt"), (log1p, expm1, "log1p"), (box_cox, cox_box, "box_cox"))
    val qofB = scala.collection.mutable.ArrayBuffer[scalation.mathstat.VectorD]()

    for f <- xform do
        if f._3 != "box_cox" then
            banner (s"$title: TranRegression with ${f._3} transform")
            val mod = new modeling.TranRegression (ox, y, ox_fname, modeling.Regression.hp, f._1, f._2)  
            model(mod, key, qofB, f._3, ox.dim2)
        else 
            for lamda <- List(0.2, 0.3, 0.4, 0.5, 0.6) do
                modeling.TranRegression.λ = lamda   
                banner (s"$title: TranRegression with ${f._3}(λ = ${lamda}) transform")
                val mod = new modeling.TranRegression (ox, y, ox_fname, modeling.Regression.hp, f._1, f._2)
                model(mod, key, qofB, f._3, ox.dim2)

    if qofB.nonEmpty then
        banner ("Copy-paste the LateX table below into a .tex file")
        val caption = s"$title: Sqrt, Log1p, and Box–Cox($$\\lambda$$ = 0.2 to 0.6)"
        val name    = title.replace(" ", "_")
        val qofs    = MatrixD(qofB.toSeq*).transpose
        val latex   = make_doc (make_table (caption, name, qofs, colName, rowName))
        println (latex)

end myTranRegression


@main def autoMPGTranRegression(): Unit =
    import modeling.Example_AutoMPG._
    val key = "val"
    myTranRegression(ox, y, ox_fname, key, "Auto MPG Regression")

@main def housePriceTranRegression(): Unit =    
    val xy = MatrixD.load ("boston_with_intercept.csv", 1, 0)
    val ox = xy(?, 0 until 8)
    val y = xy(?, 8)
    val ox_fname = Array ("_1", "x1", "x2", "x3", "x4", "x5", "x6", "x7") 
    val key = "val"  
    myTranRegression(ox, y, ox_fname, key, "House Price Regression")

@main def medicalCostTranRegression(): Unit =    
    val xy = MatrixD.load ("insurance_cat2num.csv", 1, 0)
    val ox = xy(?, 0 until 9)
    val y = xy(?, 9)

    val ox_fname = Array ("_1", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8") 
    val key = "val"  
    myTranRegression(ox, y, ox_fname, key, "Medical Cost Regression")