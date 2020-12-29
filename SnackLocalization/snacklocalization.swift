

import Foundation
import UIKit
import CoreML

let labels=["apple",
            "banana",
            "cake",
            "candy",
            "carrot",
            "cookie",
            "doughnut",
            "grape",
            "hot dog",
            "ice cream",
            "juice",
            "muffin",
            "orange",
            "pineapple",
            "popcorn",
            "pretzel",
            "salad",
            "strawberry",
            "waffle",
            "watermelon"]

struct localout
{
    let classIndex:Int
    let rect:CGRect
}

class SnackLocalization
{
    let model:snack_localization={
         do{
             let config=MLModelConfiguration()
             return try snack_localization(configuration:config)
         }catch {
             print(error)
             fatalError("Couldn't create model")
         }
     }()
    
    let fixw=224
    let fixh=224
    
    public init()
    {}
    
    public func runthemodel(image:CVPixelBuffer) -> localout?
    {
       do
       {
           let output = try model.prediction(image: image)
           var index = -1
           var minconfidence = 0.0
           for i in 0..<20
           {
               if output.output1[i].doubleValue > minconfidence
               {
                   minconfidence = output.output1[i].doubleValue
                   index = i
               }
           }
           let rect = output.output2
           let rectx=rect[0].doubleValue * Double(fixw)
           let recty=rect[2].doubleValue * Double(fixh)
           let rectw=(rect[1].doubleValue - rect[0].doubleValue) * Double(fixw)
           let recth=(rect[3].doubleValue - rect[2].doubleValue) * Double(fixh)
           return localout(classIndex: index ,
                             rect: CGRect(
                               x: rectx,
                               y: recty,
                               width: rectw,
                               height: recth))
       }
       catch
       {
           print(error)
           return nil
       }
    }
    
    
}



