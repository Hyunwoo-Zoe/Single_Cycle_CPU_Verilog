module V2to4dec(i0, i1, en, y0, y1, y2, y3);
 input i0, i1, en;
 output y0, y1, y2, y3;
 wire noti0, noti1;

 not U1(noti0, i0); // not gate를 instance화 시켜서 U1이라는 이름을 붙여줌
 not U2(noti1, i1);

 and U3(y0, noti0, noti1, en);
 and U4(y1, i0, noti1, en);
 and U5(y2, noti0, i1, en);
 and U6(y3, i0, i1, en);

 endmodule