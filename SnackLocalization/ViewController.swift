import UIKit
import CoreMedia
import CoreML
import Vision

class Rect
{
  let shapeLayer: CAShapeLayer
  let textLayer: CATextLayer

  init() {
    shapeLayer = CAShapeLayer()
    shapeLayer.fillColor = UIColor.clear.cgColor
    shapeLayer.lineWidth = 4
    shapeLayer.isHidden = true

    textLayer = CATextLayer()
    textLayer.foregroundColor = UIColor.black.cgColor
    textLayer.isHidden = true
    textLayer.contentsScale = UIScreen.main.scale
    textLayer.fontSize = 14
    textLayer.font = UIFont(name: "Avenir", size: textLayer.fontSize)
    textLayer.alignmentMode = CATextLayerAlignmentMode.center
  }

  func addToLayer(_ parent: CALayer) {
    parent.addSublayer(shapeLayer)
    parent.addSublayer(textLayer)
  }

  func show(frame: CGRect, label: String, color: UIColor) {
    CATransaction.setDisableActions(true)

    let path = UIBezierPath(rect: frame)
    shapeLayer.path = path.cgPath
    shapeLayer.strokeColor = color.cgColor
    shapeLayer.isHidden = false

    textLayer.string = label
    textLayer.backgroundColor = color.cgColor
    textLayer.isHidden = false

    let attributes = [
      NSAttributedString.Key.font: textLayer.font as Any
    ]

    let textRect = label.boundingRect(with: CGSize(width: 400, height: 100),
                                      options: .truncatesLastVisibleLine,
                                      attributes: attributes, context: nil)
    let textSize = CGSize(width: textRect.width + 12, height: textRect.height)
    let textOrigin = CGPoint(x: frame.origin.x - 2, y: frame.origin.y - textSize.height)
    textLayer.frame = CGRect(origin: textOrigin, size: textSize)
  }

  func hide() {
    shapeLayer.isHidden = true
    textLayer.isHidden = true
  }
}

class ViewController: UIViewController {

    @IBOutlet weak var videoView: UIView!
    @IBOutlet weak var resultLabel: UILabel!
    @IBOutlet weak var confidenceLabel: UILabel!
    
    let modelInputWidth : CGFloat = 224
    let modelInputHeight: CGFloat = 224
    let labels = ["apple",
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
    
    var colors: [UIColor] = []
    var boundingBox = Rect()
    
    var videoCapturer: VideoCapture!
    let semphore = DispatchSemaphore(value: ViewController.maxInflightBuffer)
    var inflightBuffer = 0
    static let maxInflightBuffer = 2
    
    lazy var classificationRequest: VNCoreMLRequest = {
        do{
            let classifier = try snack_localization(configuration: MLModelConfiguration())
            let model = try VNCoreMLModel(for: classifier.model)
            let request = VNCoreMLRequest(model: model, completionHandler: {
                [weak self] request,error in
                self?.processObservations(for: request, error: error)
            })
            request.imageCropAndScaleOption = .scaleFill
            return request
            
            
        } catch {
            fatalError("Failed to create request")
        }
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        self.setUpBoundingBoxColors()
        self.setUpCamera()
    }
    
    func setUpBoundingBoxColors() {
        for r: CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
            for g: CGFloat in [0.3, 0.7] {
                for b: CGFloat in [0.4, 0.8] {
                  let color = UIColor(red: r, green: g, blue: b, alpha: 1)
                  colors.append(color)
                }
            }
        }
    }
    
    func setUpCamera() {
        self.videoCapturer = VideoCapture()
        self.videoCapturer.delegate = self
        
        videoCapturer.frameInterval = 1
        videoCapturer.setUp(sessionPreset: .high, completion: {
            success in
            if success {
                if let previewLayer = self.videoCapturer.previewLayer {
                    self.videoView.layer.addSublayer(previewLayer)
                    self.videoCapturer.previewLayer?.frame = self.videoView.bounds
                    self.boundingBox.addToLayer(self.videoView.layer)
                    self.videoCapturer.start()
                }
            }
            else {
                print("Video capturer set up failed")
            }
        })
    }
    
}

extension ViewController: VideoCaptureDelegate {
    func videoCapture(capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
        self.localization(sampleBuffer: sampleBuffer)
    }
}


extension ViewController {
    func localization(sampleBuffer: CMSampleBuffer) {
        if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            semphore.wait()
            inflightBuffer += 1
            if inflightBuffer >= ViewController.maxInflightBuffer {
                inflightBuffer = 0
            }
            // print(CVPixelBufferGetWidth(pixelBuffer), CVPixelBufferGetHeight(pixelBuffer))
            DispatchQueue.main.async {
                let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
                do {
                    try handler.perform([self.classificationRequest])
                } catch {
                    print("Failed to perform classification: \(error)")
                }
                self.semphore.signal()
            }
            
        } else {
            print("Create pixel buffer failed")
        }
    }
}

extension ViewController {
    func processObservations(for request: VNRequest, error: Error?) {
        if let results = request.results as? [VNCoreMLFeatureValueObservation] {
            if results.isEmpty {
                self.resultLabel.text = "Nothing found"
            } else {
                guard let confidenceArray = results[0].featureValue.multiArrayValue else {
                    print("error in confidence array")
                    return
                }
                guard let boundingBoxRect = results[1].featureValue.multiArrayValue else {
                    print("error in bbox array")
                    return
                }
                
                var maxConfidenceIndex = 0
                for i in 1..<confidenceArray.count {
                    if confidenceArray[maxConfidenceIndex].compare(confidenceArray[i]) == .orderedAscending {
                        maxConfidenceIndex = i
                    }
                }
                self.resultLabel.text = self.labels[maxConfidenceIndex]
                self.confidenceLabel.text = String(format: "%.1f%%", confidenceArray[maxConfidenceIndex].doubleValue * 100)
                
                let viewWidth : CGFloat = self.videoView.bounds.width
                let viewHeight: CGFloat = viewWidth * 16 / 9  // self.videoView.bounds.height
                
                let left = viewWidth * CGFloat(boundingBoxRect[0].doubleValue)
                let right = viewWidth * CGFloat(boundingBoxRect[1].doubleValue)
                let top = viewHeight * CGFloat(boundingBoxRect[2].doubleValue)
                let bottom = viewHeight * CGFloat(boundingBoxRect[3].doubleValue)
                let rect = CGRect(x: left, y: top, width: right - left, height: bottom - top)
                self.boundingBox.show(frame: rect, label: self.labels[maxConfidenceIndex], color: self.colors[maxConfidenceIndex])
            }
        } else if let error = error {
            self.resultLabel.text = "Error: \(error.localizedDescription)"
        } else {
            self.resultLabel.text = "???"
        }
    }
}
