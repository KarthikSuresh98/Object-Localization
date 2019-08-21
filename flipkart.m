myFolder = 'E:\images';
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end
filePattern = fullfile(myFolder, '*.png');
pngFiles = dir(filePattern);
for w = 1:length(pngFiles)
  baseFileName = pngFiles(w).name;
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  i = imread(fullFileName);
  disp(size(i))
  i=rgb2gray(i);
  y=fft2(i);
  j=fftshift(y);
  disp(size(j))
  z=ones(480,640);
  x=240;
  y=320;
  d=5;
  for t=1:480
      for q=1:640
          if ((x-t)^2+(y-q)^2)^0.5<d
              z(t,q)=0;
          end
      end
  end
  s=j.*z;
  o=abs(ifft2(s));
  w=mat2gray(o);
  imwrite(w , fullFileName);
end







