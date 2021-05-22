# video-caption

## Dependencies
requirements.txt

## Dataset
[Vatex Dataset](https://eric-xw.github.io/vatex-website/download.html)

## Code Execution
For training and prediction we are using training_vatex.py

In training_vatex.py, there are three main functions: ```main()``` , ```main1()```

For training the model in the code training_vatex.py change if main block to :
```
if __name__ == '__main__':
	main()
```

After training is done it saves the model in file model.h5 saved in the current directory

Once that is done prediction can be done by executing the following steps

For prediction change the if main block to :
```
if __name__ == '__main__':
	main1()
```