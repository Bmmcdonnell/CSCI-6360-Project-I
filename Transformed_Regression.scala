package scalation
package modeling
import scalation.mathstat._

import scala.math.{sqrt, log1p, expm1}
@main def myTransRegression(): Unit =
    import Example_AutoMPG._
    val xform = List((sqrt, sq, "sqrt"), (log1p, expm1, "log1p"))

    for f <- xform do
        banner (s"TranRegression with ${f._3} transform")
        val mod = new TranRegression (ox, y, ox_fname, Regression.hp, f._1, f._2)
        mod.trainNtest ()()                                            // train and test the model
        println (mod.summary ())                                       // parameter/coefficient statistics

        banner ("Validation Test")
        mod.validate ()()

    import TranRegression.{box_cox, cox_box}
    val f = (box_cox, cox_box, "box_cox")

    for lamda <- List(0.2, 0.3, 0.4, 0.5, 0.6) do
        TranRegression.λ = lamda
        banner (s"TranRegression with ${f._3} transform")
        val mod = new TranRegression (ox, y, ox_fname, Regression.hp, f._1, f._2)
        println(TranRegression.λ)

        mod.trainNtest ()()                                            // train and test the model
        println (mod.summary ())                                       // parameter/coefficient statistics

        banner ("Validation Test")
        mod.validate ()()

end myTransRegression

@main def myTransRegression2(): Unit =
    
    val xy = MatrixD.load ("boston_with_intercept.csv", 1, 0)
    banner ("Boston House Prices")

    val ox = xy(?, 0 until 8)
    val y = xy(?, 8)
    val ox_fname = Array ("_1", "x1", "x2", "x3", "x4", "x5", "x6", "x7")                               

    val xform = List((sqrt, sq, "sqrt"), (log1p, expm1, "log1p"))

    for f <- xform do
        banner (s"TranRegression with ${f._3} transform")
        val mod = new TranRegression (ox, y, ox_fname, Regression.hp, f._1, f._2)
        mod.trainNtest ()()                                            // train and test the model
        println (mod.summary ())                                       // parameter/coefficient statistics

        banner ("Validation Test")
        mod.validate ()()

    import TranRegression.{box_cox, cox_box}
    val f = (box_cox, cox_box, "box_cox")

    for lamda <- List(0.2, 0.3, 0.4, 0.5, 0.6) do
        TranRegression.λ = lamda
        banner (s"TranRegression with ${f._3} transform")
        val mod = new TranRegression (ox, y, ox_fname, Regression.hp, f._1, f._2)
        println(TranRegression.λ)

        mod.trainNtest ()()                                            // train and test the model
        println (mod.summary ())                                       // parameter/coefficient statistics

        banner ("Validation Test")
        mod.validate ()()
    
end myTransRegression2

