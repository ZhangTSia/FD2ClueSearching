function DrawHeadMesh(vertex, tri, keyPoints, normal)

trisurf(tri', vertex(1, :), vertex(2, :), vertex(3, :));

if nargin >= 3
    hold on
    plot3(keyPoints(1, :), keyPoints(2, :), keyPoints(3, :), 'm.', 'MarkerSize',15)
end

if nargin == 4
    hold on
    quiver3(vertex(1, :), vertex(2, :), vertex(3, :), normal(1, :), normal(2, :), normal(3, :))
end

axis equal
axis vis3d
%axis([0 250 0 250])
%axis([12 270 -8 270])
%alpha(0.5);
title('standard head mesh');
xlabel('x');
ylabel('y');
zlabel('z');