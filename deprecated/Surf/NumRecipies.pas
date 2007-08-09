UNIT NumRecipies;
INTERFACE

USES Dialogs,SysUtils;

CONST
  NP     = 6;  // number of parameters
  NDATAP = 16; // number of data points

  //simplex
  MP     = NP + 1;
  MAXERR = 1000000;
  MAXDIM = MP;
TYPE
  DArrayNP      = array[1..NP] of DOUBLE;
  DArrayNPbyNP  = array[1..NP] of DArrayNP;
  IArrayNP      = array[1..NP] of integer;
  DArrayNDATA   = array[1..NDATAP] of DOUBLE; // for levenberg-marquadt

  //simplex
  DArrayMP      = array[1..MP] of DOUBLE;
  DArrayMPbyNP  = array[1..MP] of DArrayNP;
  DArrayNPbyMP  = array[1..NP] of DArrayMP;

  //DArrayNPby1  = array[1..NP,1..1] of DOUBLE; // for levenberg-marquadt

  LevMarqObj = class
    public
      { Public declarations }
      Mrqmin0chisq : double;
      MrqminBeta   : DArrayNP;

      Procedure CheckBounds(var a : DArrayNP);virtual; abstract;
      Procedure func(xx : double;
                  var a : DArrayNP;
               var yfit : double;
               var dyda : DarrayNP;
                ma,mfit : integer;
            computedyda : boolean); virtual; abstract;
      //Procedure CallOut(iter : integer; var yfit : DArrayNDATA); virtual; abstract;
      procedure mrqmin(var x,y,sig     : DArrayNDATA;
                       ndata           : integer;
                       var a           : DArrayNP;
                       ma              : integer;
                       var lista       : IArrayNP;
                       mfit            : integer;
                       var covar,alpha : DArrayNPbyNP;
                       var chisq,alamda: double);
    private
      { Private declarations }
      Procedure mrqcof(var x,y,sig : DArrayNDATA;
                       var a : DArrayNP;
                   var lista : IArrayNP;
                   var alpha : DArrayNPbyNP;
                    var beta : DArrayNP;
                   var chisq : double;
               mfit,ndata,ma : integer);
  end;

  CSimplex = class
    public
      { Public declarations }
      ndim,best,index : integer;
      ftol : double;
      iter : integer;
      verticies : DArrayMPbyNP;
      funcvals  : DArrayMP;
      function func(var pr : DArrayNP) : double; virtual; abstract;
      procedure callout(var pr : DArrayNP); virtual; abstract;
      procedure Run;
    private
      { Private declarations }
      mpts,nextworst,worst: integer;
      ytry,ysave,sum,rtol,den : double;
      psum : DArrayNP;
      function Amotry(fac : double): double;
      Procedure RankVertecies;
  end;

  Function sign(a,b : double) : double;
  Function pythag(a,b : double) : double;
  Procedure tred2(var a : DArrayNPbyNP;
                      n : integer;
                var d,e : DArrayNP);
  Procedure tred2for2(var a : DArrayNPbyNP;
                    var d,e : DArrayNP);
  Procedure tqli(var d,e : DArrayNP;
                       n : integer;
                   var z : DArrayNPbyNP);
  Procedure gaussj(VAR a : DArrayNPbyNP;
                       n : integer;
                   VAR b : DArrayNPbyMP;
                       m : integer);
  Procedure jacobi(VAR a : DArrayNPbyNP;
                       n : integer;
                   VAR d : DArrayNP;
                   VAR v : DArrayNPbyNP;
                VAR nrot : integer);

IMPLEMENTATION

function CSimplex.Amotry;
{Extrapolates by a factor fac through the face of the simplex across from
the high point, tries, it, and replaces the high point if the new point
is better.}
var j : integer;
    fac1,fac2,ytry : double;
    ptry : DArrayNP;
begin
  fac1 := (1.0-fac)/ndim;
  fac2 := fac1-fac;
  For j := 1 to ndim do
    ptry[j] := psum[j]*fac1-verticies[worst,j]*fac2;
  ytry := func(ptry); //Evaulate the function at the trial point
  iter := iter + 1;
  If ytry < funcvals[worst] then //If its better than the highest, then replace the highest
  begin
    funcvals[worst] := ytry;
    For j:=1 to ndim do
    begin
      psum[j] := psum[j] + ptry[j] - verticies[worst,j];
      verticies[worst,j] := ptry[j];
    end;
  end;
  amotry := ytry;
end;

{==============================================================================}
Procedure CSimplex.RankVertecies;
var i : integer;
begin
  best := 1;
  //First, we must determine which point is the higest(worst),
  // next highest, and lowest (best),
  if funcvals[1] > funcvals[2] then
  begin
    worst := 1;
    nextworst := 2;
  end else
  begin
    worst := 2;
    nextworst := 1;
  end;
  //by looping over the points in the simplex
  For i := 1 to mpts do
  begin
    if funcvals[i] < funcvals[best] then best := i;
    if funcvals[i] > funcvals[worst] then
    begin
      nextworst := worst;
      worst := i;
    end
    else if funcvals[i] > funcvals[nextworst] then
           if i <> worst then nextworst := i;
  end;
end;

{==============================================================================}
Procedure CSimplex.Run;
{Multidimensional minimization of the function func(x) where x[1..ndim] is a
vector in ndim dimensions, by the downhill simplex method of Nelder and Mead.
The matrix p[1..ndim+1][1..ndim] is input.  Its ndim+1 rows are
ndim-dimensional vectors which are the vertices of the starting simplex.
Also input is the vector y[1..ndim+1], whose components must be
pre-intialized to the values of func evaluated at the ndim+1 vertices (rows)
of p; and ftol the fractional convergence tolerance to be achieved in the
function value (n.b.!). On output, p and y will have been reset to ndim+1 new
points all within ftol of a minimum function value, and iter gives the
number of function evaluations taken.}
CONST
  itermax = 1000; //the maximum allowed number of function evaluations
  alpha = 1.0{default=1.0};{reflection}
  beta = 0.5 {default=0.5};{contraction}
  gamma = 2.0{default=2.0};{expansion}
var i,j : integer;
begin
  if ndim > MAXDIM then
  begin
    ShowMessage('Number of dimensions ('+IntToStr(ndim)+') exceeds maximum allowed('+IntToStr(MAXDIM)+')');
    exit;
  end;
  mpts := ndim + 1;
  iter := 0;
  For j := 1 to ndim do
  begin
    sum := 0.0;
    for i := 1 to mpts do
      sum := sum + verticies[i,j];
    psum[j] := sum;
  end;
  While TRUE do
  begin
    RankVertecies;
    //compute the fractional range from higest to lowest
    den := (abs(funcvals[worst])+abs(funcvals[best]));
    if den <> 0
      then rtol := 2.0*abs(funcvals[worst]-funcvals[best])/den
      else rtol := 0;
    if rtol < ftol then exit; //return if satisfactory
    if iter >= itermax then  exit;//return if run out of iterations
    //Begin a new iteration.  First extrapolate by a factor of alpha through the face
    //of the simplex across from the high point, i.e., reflect the simplex from the high point.
    ytry := amotry(-alpha);
    if ytry <= funcvals[best] then//gives a better result than the best point, so try
    begin                         //an additional extrapolation by a factor of gamma
      CallOut(verticies[best]);{this calls a user defined procedure to do something with the information}
      ytry := amotry(gamma);
    end else if ytry >= funcvals[nextworst] then
      begin //the reflected point is worse than the second highest, so look for an
            //interpmediate lower point, i.e., do a one-dimensional contraction
        ysave := funcvals[worst];
        ytry := amotry(beta);
        if ytry >= ysave then  //can't seem to get rid of that high point. Better
        begin                  //contract around the lowest(best) point --SHRINK--
          for i := 1 to mpts do
            if i <> best then
            begin
              for j := 1 to ndim do
              begin
                psum[j] := 0.5 * (verticies[i,j]+verticies[best,j]);
                verticies[i,j] := psum[j];
              end;
              funcvals[i] := func(psum);
            end;
          iter := iter + ndim;  //keep track of function evaluations
          For j := 1 to ndim do //recompute psum
          begin
            sum := 0.0;
            For i := 1 to mpts do
              sum := sum + verticies[i,j];
            psum[j] := sum;
          end;
        end;
      end;
  end;  //go back for the test of doneness and the next iteration.
end;


{ ======================================================================= }
Function sign(a,b : double) : double;
{ make the sign of _a_ equal to the sign of _b_ }
begin
  if b < 0 then sign := -abs(a) else sign := abs(a);
end;

{ ======================================================================= }
Function pythag(a,b : double) : double;
{Computes sqrt(a2 + b2) without destructive underflow or overflow}
var absa,absb : double;
begin
  absa := abs(a);
  absb := abs(b);
  if (absa > absb)
    then pythag := absa * sqrt(1.0 + sqr(absb/absa))
    else pythag := absb * sqrt(1.0 + sqr(absa/absb));
end;

{ ======================================================================= }
Procedure tred2(var a : DArrayNPbyNP;
                    n : integer;
              var d,e : DArrayNP);
{ From Numerical recipies in PASCAL}
{ Householder reduction of a real, symmetric matrix a.  On output, a is
  replaced by the orthogonal matrix Q effecting the transformation. d returns
  the diagonal elements of the tridiagonal matrix, and e the off-diagonal
  elements, with e[1] = 0.  Several statements, as noted in comments, can be
  omitted if only eigenvalues are to be found, in which case a contains no
  useful information on output.  Otherwise they are to be included.
}
var l,k,j,i : integer;
    scale,hh,h,g,f : double;
begin
  for i := n downto 2 do
  begin
    l     := i-1;
    h     := 0.0;
    scale := 0.0;
    if l > 1 then
    begin
      for k := 1 to l do scale := scale + abs(a[i,k]);
      if scale = 0.0 then e[i] := a[i,l] {Skip transformation}
      else
      begin {transformation}
        for k := 1 to l do
        begin
          a[i,k] := a[i,k] / scale;  {Use scaled a's for transformation}
          h := h + a[i,k]*a[i,k];    {Form sigma in h}
        end;
        f    := a[i,l];
        g    := -sign(sqrt(h),f);
        e[i] := scale*g;
        h    := h - f*g;
        a[i,l] := f-g;               {Store u in the ith row of a}
        f    := 0.0;
        for j := 1 to l do
        begin
          {Next statement can be omitted if eigenvectors not wanted}
          a[j,i] := a[i,j]/h;        {Store u/H in ith column of a}
          g := 0.0;                  {Form an element of A.u in g}
          for k := 1 to j do g := g + a[j,k]*a[i,k];
          for k := j+1 to l do g := g + a[k,j]*a[i,k];
          e[j] := g/h;               {Form element of p in temporarily unused element of e}
          f := f + e[j]*a[i,j];
        end;
        hh := f/(h+h);               {Form K}
        for j := 1 to l do           {Form q and store in e overwriting p}
        begin
          f    := a[i,j];            {Note that e[l]=e[i-1] survives}
          g    := e[j]-hh*f;
          e[j] := g;
          for k := 1 to j do         {Reduce a}
            a[j,k] := a[j,k] - f*e[k] -g*a[i,k];
        end;
      end;
    end else e[i] := a[i,l];
    d[i] := h;
  end;
  {Next statement can be omitted if eigenvectors not wanted}
  d[1]:=0.0;
  e[1]:=0.0;
  {Contents of this loop can be omitted if eigenvectors not wanted
   except for statement d[i]:=a[i,i];}
  for i := 1 to n do
  begin                              {Begin accumulation of transformation matricies}
    l := i-1;
    if d[i] <> 0 then                {This block skipped when i=1}
    begin
      for j := 1 to l do
      begin
        g := 0.0;
        for k := 1 to l do
          g := g + a[i,k]*a[k,j];    {Use u and u/H stored in a ti form P.Q}
        for k := 1 to l do
          a[k,j] := a[k,j] - g*a[k,i];
      end;
    end;
    d[i] := a[i,i];                  {This statement remains}
    a[i,i] := 1.0;                   {Reset row and column of a to identity}
    for j := 1 to l do               {matrix for next iteration}
    begin
      a[i,j] := 0.0;
      a[j,i] := 0.0;
    end;
  end;
end;

{ ======================================================================= }
Procedure tred2for2(var a : DArrayNPbyNP;
                  var d,e : DArrayNP);
{ From Numerical recipies in PASCAL}
{ Householder reduction of a real, symmetric matrix a.  On output, a is
  replaced by the orthogonal matrix Q effecting the transformation. d returns
  the diagonal elements of the tridiagonal matrix, and e the off-diagonal
  elements, with e[1] = 0.  Several statements, as noted in comments, can be
  omitted if only eigenvalues are to be found, in which case a contains no
  useful information on output.  Otherwise they are to be included.
}
var k,i,j,l : integer;
    {scale,h,hh,}g{,f} : double;
begin
{  i     := 2;
  l     := i-1;
  h     := 0.0;
  scale := 0.0;
}  d[2]  := 0.0;
  e[2]  := a[2,1];

  d[1]:=0.0;
  e[1]:=0.0;
  for i := 1 to 2 do
  begin                              {Begin accumulation of transformation matricies}
    l := i-1;
    if d[i] <> 0 then                {This block skipped when i=1}
    begin
      for j := 1 to l do
      begin
        g := 0.0;
        for k := 1 to l do
          g := g + a[i,k]*a[k,j];    {Use u and u/H stored in a ti form P.Q}
        for k := 1 to l do
          a[k,j] := a[k,j] - g*a[k,i];
      end;
    end;
    d[i] := a[i,i];                  {This statement remains}
    a[i,i] := 1.0;                   {Reset row and column of a to identity}
    for j := 1 to l do               {matrix for next iteration}
    begin
      a[i,j] := 0.0;
      a[j,i] := 0.0;
    end;
  end;
end;

{ ======================================================================= }
Procedure tqli(var d,e : DArrayNP;
                     n : integer;
                 var z : DArrayNPbyNP);
{ From Numerical recipies in PASCAL}
{ QL algorithm with implicit shifts, to determine the eigenvalues and eigenvectors
  of a real, symmetric, tridiagonal matrix, or of a real, symmetric matrix
  previously reduced by tred2.  On input, d contains the diagonal elements of the
  tridiagonal matrix.  On output, it returns the eigenvalues.  The vector e inputs
  the subdiagonal elements of the tridiagonal matrix, with e[1] arbitrary.  On
  output, e is destroyed.  When finding only the eigenvalues, several lines may be
  omitted, as noted in the comments.  If the eigenvectors of a tridiagonal matrix
  are desired, the matrix z is input as the identity matrix.  If the eigenvectors
  of a matrix that has been reduced by tred2 are required, then z is input as the
  matrix output by tred2.  In either case, the kth column of z returns the
  normalized eigenvector corresponding to d.
}
label 10,20;

var m,l,iter,i,k  : integer;
    s,r,p,g,f,dd,c,b : double;
begin
  for i := 2 to n do e[i-1] := e[i]; {Convenient to renumber the elements of e}
  e[n] := 0.0;
  for l := 1 to n do
  begin
    iter := 0;
10: for m := l to n-1 do   { Look for a single small subdiagonal element}
    begin                  { to split the matrix}
      dd := abs(d[m])+abs(d[m+1]);
      if abs(e[m]) + dd = dd then goto 20;
    end;
    m := n;
20: if m <> l then
    begin
      if iter = 30 then
        ShowMessage('Error in QTLI, iter='+inttostr(iter));
      iter := iter + 1;
      g := (d[l+1]-d[l])/(2.0*e[l]); {Form shift}
      r := sqrt(sqr(g)+1.0);
      g := d[m]-d[l]+e[l]/(g+sign(r,g)); {This is dm - ks}
      s := 1.0;
      c := 1.0;
      p := 0.0;
      for i := m - 1 downto l do
      begin
        f := s*e[i];
        b := c*e[i];

        {original pascal}
        if abs(f) >= abs(g) then
        begin
          c := g/f;
          r := sqrt(sqr(c)+1.0);
          e[i+1] := f*r;
          s := 1.0/r;
          c := c*s;
        end else
        begin
          s := f/g;
          r := sqrt(sqr(s)+1.0);
          e[i+1] := g*r;
          c := 1.0/r;
          s := s*c;
        end;
        g := d[i+1]-p;
        r := (d[i]-g)*s+2.0*c*b;
        p := s*r;
        d[i+1] := g+p;
        g := c*r-b;

        {Next loop can be omitted if eigenvectors not wanted}
        for k := 1 to n do
        begin
          f := z[k,i+1];
          z[k,i+1] := s*z[k,i]+c*f;
          z[k,i] := c*z[k,i]-s*f;
        end;
      end;
      d[l] := d[l] - p;
      e[l] := g;
      e[m] := 0.0;
      goto 10;
    end;
  end;
end;

{ ======================================================================= }
Procedure jacobi(VAR a : DArrayNPbyNP;
                     n : integer;
                 VAR d : DArrayNP;
                 VAR v : DArrayNPbyNP;
              VAR nrot : integer);
{Computes all eigenvalues and eigenvectors of a real symmetric matrix
 z[1..n][1..n].  On output, elements of a above the diagonal are destroyed.
 d[1..n] returns the eigenvalues of a.  v[1..n][1..n] is a matrix whose
 columns contain, on output, the normalized eigenvectors of a.  nrot returns
 the number of Jacobi rotations that were required.}
Label 99;
var
  j,iq,ip,i : integer;
  tresh,theta,tau,t,sm,s,h,g,c : double;
  b,z: ^DArrayNP;
begin
  new(b);
  new(z);
  For ip := 1 to n do
  begin
    for iq := 1 to n do v[ip,iq] := 0.0; {initialize the identity matrix}
    v[ip,ip] := 1.0;
  end;
  For ip := 1 to n do
  begin
    b^[ip] := a[ip,ip];
    d[ip] := b^[ip];
    z^[ip] := 0.0;
  end;
  nrot := 0;
  For i := 1 to 50 do
  begin
    sm := 0.0;
    For ip := 1 to n-1 do
      for iq := ip+1 to n do
        sm := sm+abs(a[ip,iq]);
    if sm = 0.0 then goto 99;
    if i < 4
      then tresh := 0.2*sm/sqr(n)
      else tresh := 0.0;
    for ip := 1 to n-1 do
    begin
      for iq := ip+1 to n do
      begin
        g := 100.0*abs(a[ip,iq]);
        if (i>4) and (abs(d[ip])+g = abs(d[ip]))
          and (abs(d[iq])+g = abs(d[iq])) then a[ip,iq] := 0.0
        else if abs(a[ip,iq]) > tresh then
        begin
          h := d[iq]-d[ip];
          if abs(h)+g = abs(h) then t := a[ip,iq]/h
          else
          begin
            theta := 0.5*h/a[ip,iq];
            t := 1.0/(abs(theta)+sqrt(1.0+sqr(theta)));
            if theta < 0.0 then t := -t;
          end;
          c := 1.0/sqrt(1+sqr(t));
          s := t*c;
          tau := s/(1.0 + c);
          h := t*a[ip,iq];
          z^[ip] := z^[ip]-h;
          z^[iq] := z^[iq]+h;
          d[ip] := d[ip]-h;
          d[iq] := d[iq]+h;
          a[ip,iq] := 0.0;
          for j := 1 to ip-1 do
          begin
            g := a[j,ip];
            h := a[j,iq];
            a[j,ip] := g-s*(h+g*tau);
            a[j,iq] := h+s*(g-h*tau);
          end;
          for j := ip+1 to iq-1 do
          begin
            g := a[ip,j];
            h := a[j,iq];
            a[ip,j] := g-s*(h+g*tau);
            a[j,iq] := h+s*(g-h*tau);
          end;
          for j := iq+1 to n do
          begin
            g := a[ip,j];
            h := a[iq,j];
            a[ip,j] := g-s*(h+g*tau);
            a[iq,j] := h+s*(g-h*tau);
          end;
          for j := 1 to n do
          begin
            g := v[j,ip];
            h := v[j,iq];
            v[j,ip] := g-s*(h+g*tau);
            v[j,iq] := h+s*(g-h*tau);
          end;
          nrot := nrot + 1;
        end;
      end;
    end;
    for ip := 1 to n do
    begin
      b^[ip] := b^[ip] + z^[ip];
      d[ip] := b^[ip];
      z^[ip] := 0.0;
    end;
  end;
  ShowMessage('Pause in jacobi routine: 50 iterations should not happen.');
99:
  dispose(z);
  dispose(b);
end;

{ ======================================================================= }
Procedure gaussj(VAR a : DArrayNPbyNP;
                     n : integer;
                 VAR b : DArrayNPbyMP;
                     m : integer);
{Linear equation solution by Gauss-Jordan elimination.  The input matrix
 a[1..n,1..n] has n by n elements. b[1..n,1..m] is an input matrix of size
 n by m containing the m right-hand side vectors.  On output, a is replaced
 by its matrix inverse, and b is replaced by the corresponding solution
 vectors.}
const tiny = 1.0e-10;
var
  big,dum,pivinv : double;
  i,icol,irow,j,k,l,ll : integer;
  indxc,indxr,ipiv: ^IArrayNP;
begin
  new(indxc);
  new(indxr);
  new(ipiv);
  icol := 0;
  irow := 0;
  for j := 1 to n do ipiv^[j] := 0;
  for i := 1 to n do {this is the main loop over the columns to be reduced}
  begin
    big := 0.0;
    for j := 1 to n do
      if ipiv^[j] <> 1 then
        for k := 1 to n do
          if ipiv^[k] = 0 then
            if abs(a[j,k]) >= big then
            begin
              big := abs(a[j,k]);
              irow := j;
              icol := k;
            end else if ipiv^[k] > 1 then
              ShowMessage('Pause in GAUSSJ - singular matrix');
    ipiv^[icol] := ipiv^[icol] + 1;
    if irow <> icol then
    begin
      for l := 1 to n do
      begin
        dum := a[irow,l];
        a[irow,l] := a[icol,l];
        a[icol,l] := dum;
      end;
      for l := 1 to m do
      begin
        dum := b[irow,l];
        b[irow,l] := b[icol,l];
        b[icol,l] := dum;
      end;
    end;
    indxr^[i] := irow;
    indxc^[i] := icol;
    if abs(a[icol,icol]) < tiny then
    begin
      //ShowMessage('Pause 2 in GAUSSJ - singular matrix');
      //beep;
      a[icol,icol] := tiny;
    end;

    pivinv := 1.0/a[icol,icol];
    a[icol,icol] := 1.0;
    for l := 1 to n do
      a[icol,l] := a[icol,l] * pivinv;
    for l := 1 to m do
      b[icol,l] := b[icol,l] * pivinv;
    for ll := 1 to n do
      if ll <> icol then
      begin
        dum := a[ll,icol];
        a[ll,icol] := 0.0;
        for l := 1 to n do
          a[ll,l] := a[ll,l]-a[icol,l]*dum;
        for l := 1 to m do
          b[ll,l] := b[ll,l]-b[icol,l]*dum;
      end;
  end;

  for l := n downto 1 do
    if indxr^[l] <> indxc^[l] then
      for k := 1 to n do
      begin
        dum := a[k,indxr^[l]];
        a[k,indxr^[l]] := a[k,indxc^[l]];
        a[k,indxc^[l]] := dum;
      end;

  dispose(indxc);
  dispose(indxr);
  dispose(ipiv);
end;

{ ======================================================================= }
Procedure invertmatrix(VAR a : DArrayNPbyNP;
                           n : integer);
var x : integer;
    i : DArrayNPbyMP;
begin
  FillChar(i,sizeof(i),0);
  For x := 1 to n do i[x,x] := 1.0;

  gaussj(a,n,i,n);
end;

{ ======================================================================= }
Procedure sqrmatrixmult(a,b : DArrayNPbyNP;
                          n : integer;
                      VAR c : DArrayNPbyNP);
{ Multiply matrix a by matrix b, put result in matrix c.  Assumes
  all are nxn square matricies}
var z,row,col : integer;
begin
  For row := 1 to n do
    For col := 1 to n do
    begin
      c[row,col] := 0.0;
      For z := 1 to n do
        c[row,col] := c[row,col] + a[row,z] * b[z,col];
    end;
end;

{ ======================================================================= }
Procedure covsrt(VAR covar : DArrayNPbyNP;
                        ma : integer;
                 VAR lista : IArrayNP;
                      mfit : integer);
{Given the covariance matrix covar[1..ma,1..ma] of a fit for mfit of ma total
 parameters, and their ordering lista[1..ma], repack the covariance matrix to
 the true order of the parameters.  Elements associated with fixed parameters
 will be zero.}
var
  j,i : integer;
  swap : double;
begin
  for j := 1 to ma-1 do
    for i := j+1 to ma do covar[i,j] := 0.0;
  for i := 1 to mfit-1 do
  begin
    for j := i+1 to mfit do
      if lista[j] > lista[i]
        then covar[lista[j],lista[i]] := covar[i,j]
        else covar[lista[i],lista[j]] := covar[i,j];
  end;
  swap := covar[1,1];
  for j := 1 to ma do
  begin
    covar[1,j] := covar[j,j];
    covar[j,j] := 0.0;
  end;
  covar[lista[1],lista[1]] := swap;
  for j := 2 to mfit do
    covar[lista[j],lista[j]] := covar[1,j];
  for j := 2 to ma do
    for i := 1 to j-1 do
      covar[i,j] := covar[j,i];
end;

{ ======================================================================= }
Procedure ludcmp(VAR a : DArrayNPbyNP;
                     n : integer;
              VAR indx : IArrayNP;
                     d : double);
{Given an n x n  matrix a[1..n,1..n], this routine replaces it by the LU
 decomposition of a rowwise permutation of itself. a and n are input. a is
 output; indx[1..n] is an outout vector which records the row permutation
 affected by the partial pivoting; d is output as +/1 1 depending on whether
 the number of row interchanges was even or odd, respectively.  This routine
 is used in combination with lubksb to solve linear equations or invert a matrix.}
const tiny = 1.0e-20;
var
  k,j,imax,i : integer;
  sum,dum,big : double;
  vv : ^DArrayNP;
begin
  new(vv);
  d := 1.0;
  for i := 1 to n do
  begin
    big := 0.0;
    for j := 1 to n do
      if abs(a[i,j]) > big then big := abs(a[i,j]);
    if big = 0.0 then
    begin
      //ShowMessage('Pause in LUDCMP - singular matrix');
      big := tiny; //my addition so that it can go on
    end;
    vv^[i] := 1.0/big;
  end;
  for j := 1 to n do
  begin
    for i := 1 to j-1 do
    begin
      sum := a[i,j];
      for k := 1 to i-1 do
        sum := sum -a[i,k]*a[k,j];
      a[i,j] := sum;
    end;
    big := 0.0;
    imax := 0;
    for i := j to n do
    begin
      sum := a[i,j];
      for k := 1 to j-1 do
        sum := sum-a[i,k]*a[k,j];
      a[i,j] := sum;
      dum := vv^[i]*abs(sum);
      if dum >= big then
      begin
        big := dum;
        imax := i;
      end;
    end;
    if j <> imax then
    begin
      for k := 1 to n do
      begin
        dum := a[imax,k];
        a[imax,k] := a[j,k];
        a[j,k] := dum;
      end;
      d := -d;
      vv^[imax] := vv^[j];
    end;
    indx[j] := imax;
    if a[j,j] = 0.0 then a[j,j] := tiny;
    if j <> n then
    begin
      dum := 1.0/ a[j,j];
      for i := j+1 to n do
        a[i,j] := a[i,j]*dum;
    end;
  end;
  dispose(vv);
end;

{ ======================================================================= }
Procedure lubksb(VAR a : DArrayNPbyNP;
                     n : integer;
              VAR indx : IArrayNP;
                 VAR b : DArrayNP);
{Solves the set of n linear equations A.X = B.  here a[1..n,1..n] is input,
 not as the matrix A but rather as its LU decomposition, determined by the
 routine ludcmp.  b[1..n] is input as the right hand-side vetor, and returns
 with the solution vector X.  a,n, and indx are not modified by thie routine
 and can be left in place for successive calls with different right-hand sides b.
 Thie routine takes into account the possibility that b will begin with many
 zero elements, so it is efficient for use in matrix inversion.}
var
  j,ip,ii,i : integer;
  sum : double;
begin
  ii := 0;
  for i := 1 to n do
  begin
    ip := indx[i];
    sum := b[ip];
    b[ip] := b[i];
    if ii <> 0
      then for j := ii to i-1 do
             sum := sum - a[i,j]*b[j]
      else if sum <> 0.0 then ii := i;
    b[i] := sum;
  end;
  for i := n downto 1 do
  begin
    sum := b[i];
    for j := i+1 to n do
      sum := sum - a[i,j]*b[j];
    b[i] := sum/a[i,i];
  end;
end;

{ ======================================================================= }
Procedure ludecomp(VAR a : DArrayNPbyNP;
                       n : integer;
                   VAR b : DArrayNP;
                       m : integer);
{Linear equation solution by LU decomposition.  The input matrix
 a[1..n,1..n] has n by n elements. b[1..n,1..m] is an input matrix of size
 n by m containing the m right-hand side vectors.  On output, a is destroyed,
 and b is replaced by the corresponding solution vector.}

var indx : IArrayNP;
    d : double;
begin
  d := 0;
  ludcmp(a,n,indx,d);
  lubksb(a,n,indx,b);
end;

{==============================================================================}
Procedure LevMarqObj.mrqcof(var x,y,sig : DArrayNDATA;
                       var a : DArrayNP;
                   var lista : IArrayNP;
                   var alpha : DArrayNPbyNP;
                    var beta : DArrayNP;
                   var chisq : double;
               mfit,ndata,ma : integer);
var
  k,j,i : integer;
  ymod,wt,sig2i,dy : double;
  dyda : ^DArrayNP;
begin
  new(dyda);
  for j := 1 to mfit do
  begin
    for k := 1 to j do alpha[j,k] := 0.0;
    beta[j] := 0.0;
  end;
  chisq := 0.0;
  CheckBounds(a);
  for i := 1 to ndata do
  begin
    func(x[i],a,ymod,dyda^,ma,mfit,true);
    if sig[i] = 0.0 then ShowMessage('Divide by zero in mrqcof 1');
    sig2i := 1.0/(sig[i]*sig[i]);
    dy := y[i]-ymod;
    for j := 1 to mfit do
    begin
      wt := dyda^[lista[j]]*sig2i;
      for k := 1 to j do
        alpha[j,k] := alpha[j,k] + wt*dyda^[lista[k]];
      beta[j] := beta[j] + dy*wt;
    end;
    chisq := chisq+dy*dy*sig2i;
  end;
  for j := 2 to mfit do
    for k := 1 to j-1 do alpha[k,j] := alpha[j,k];
  dispose(dyda);
end;

{==============================================================================}
Procedure LevMarqObj.mrqmin(
                 var x,y,sig     : DArrayNDATA;
                 ndata           : integer;
                 var a           : DArrayNP;
                 ma              : integer;
                 var lista       : IArrayNP;
                 mfit            : integer;
                 var covar,alpha : DArrayNPbyNP;
                 var chisq,alamda: double);
LABEL 99;
var
  k,kk,j,ihit : integer;
  atry,da : ^DArrayNP;
  oneda : ^DArrayNP{byMP};
begin
  new(da);
  new(oneda);
  new(atry);
  if alamda < 0.0 then
  begin
    kk := mfit + 1;
    for j := 1 to ma do
    begin
      ihit := 0;
      for k := 1 to mfit do
        if lista[k] = j then ihit := ihit + 1;
      if ihit = 0 then
      begin
        lista[kk] := j;
        kk := kk+1;
      end else if ihit > 1 then
        ShowMessage('Pause 1 in routine MRQMIN: Improper permutation in LISTA');
    end;
    if kk <> ma+1 then
      ShowMessage('Pause 2 in routine MRQMIN: Improper permutation in LISTA');
    alamda := 0.001;
    mrqcof(x,y,sig,a,lista,alpha,MrqminBeta,chisq,mfit,ndata,ma);
    Mrqmin0chisq := chisq;
    for j := 1 to ma do atry^[j] := a[j];
  end;
  for j := 1 to mfit do
  begin
    for k := 1 to mfit do covar[j,k] := alpha[j,k];
    covar[j,j] := alpha[j,j]*(1.0+alamda);
    oneda^[j{,1}] := MrqminBeta[j];
  end;
  //gaussj(covar,mfit,oneda^,1);
  ludecomp(covar,mfit,oneda^,1);

  for j := 1 to mfit do
    da^[j] := oneda^[j{,1}];
  if alamda = 0.0 then
  begin
    covsrt(covar,ma,lista,mfit);
    goto 99;
  end;
  for j := 1 to mfit do
    atry^[lista[j]] := a[lista[j]]+da^[j];
  mrqcof(x,y,sig,atry^,lista,covar,da^,chisq,mfit,ndata,ma);
  if chisq < Mrqmin0chisq then
  begin
    alamda := 0.1*alamda;
    Mrqmin0chisq := chisq;
    for j := 1 to mfit do
    begin
      for k := 1 to mfit do alpha[j,k] := covar[j,k];
      MrqminBeta[j] := da^[j];
      a[lista[j]] := atry^[lista[j]];
    end;
  end else
  begin
    alamda := 10.0 * alamda;
    chisq := Mrqmin0chisq;
  end;
99 :
  dispose(atry);
  dispose(oneda);
  dispose(da);
end;

END.


