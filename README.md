## Grape_leaf_diseases_detection_application
Hey....  
Following are the steps to deploy your application over heroku platform:   
*step 1: If you don't have heroku inside your system then install it...*  
*step 2: After installing heroku make the folder name(your choice) and put the files as they are present in this repo.*  
*step 3: We carefull with the folder names (means caps should be caps and small letters should be small letters if not done your application will give error).like Procfile(P should be capital).*  
*step 4:Now, your application is ready inside the folder then open the git bash inside your application directory and write following commands their one by one.(things written after // are comments so don't write them over git bash)*  
#### 1. git init  
#### 2. git add .  
#### 3. git commit -m "write some commit message"  
#### 4. heroku create    
*//(for creating application its up to you , you can keep name here and can avoid it as well(suggested is don't write name here))*  
#### 5. git remote -v  
*//(to check where the current application will be pushed)*  
#### 6. git push heroku master  
    
Here I am also providing the steps for making your repo over github.Firstly make a repo inside your github account then copy the repo URL and then open git bash inside the folder directory.Now, write the following commands over the git bash.   
#### 1. git init  
#### 2. git add .  
#### 3. git commit -m "write the commit message of your choice"  
#### 4. git remote add origin (Now paste the copied repo URL here)
#### 5. git remote -v //(path where the repo will be pushed will be shown here it is just for checking and it should show the repo address that you pasted in previous command)  
#### 6. git push -f origin master  
Bravo .....  
your repo is successfully made(Hopefully).......
