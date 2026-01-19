假设某个策略为最优策略，在通常的定义中，这等价于任何该策略对于任何状态的状态价值函数的值都为最大

在**Policy Gradient**中，我们使用$J(\theta)$来衡量策略的优劣，一个策略的$J(\theta)$定义为在所有轨迹上的$R(\tau)$的期望：
$$
J(\theta)=\mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)]
$$
其中$\theta$代表策略函数的参数，$\tau$为一条轨迹，$R(\tau)$则是这条轨迹的总的折扣回报

则$J(\theta)$对于$\theta$的梯度为：
$$
\begin{align}
\nabla_{\theta}J(\theta)&=\nabla_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)]\\
&=\nabla_{\theta}\sum_\tau P(\tau|\theta)R(\tau)\\
&=\sum_\tau \nabla_\theta [P(\tau|\theta)R(\tau)]\\
&=\sum_\tau R(\tau) \nabla_\theta P(\tau|\theta)\\
&=\sum_\tau R(\tau)[P(\tau|\theta) \nabla_\theta \ln P(\tau|\theta)]\\
&=\sum_\tau P(\tau|\theta) R(\tau) \nabla_\theta \ln P(\tau|\theta)\\
&=\mathbb{E}_{\tau \sim \pi_\theta}[R(\tau) \nabla_\theta \ln P(\tau|\theta)] \tag{1}
\end{align}
$$
我们定义$T$为轨迹的长度，意为轨迹中动作的数量，例如轨迹$\{s_0,a_0,r_0,s_1,a_1,r_1,s_2\}$的长度$T$为$2$，则：
$$
\begin{align}
\ln P(\tau|\theta)&=\ln \left[P(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)\right]\\
&=\ln P(s_0) + \ln \left[\prod_{t=0}^{T-1} \pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)\right]\\
&=\ln P(s_0) + \sum_{t=0}^{T-1}\ln[\pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)]\\
&=\ln P(s_0) + \sum_{t=0}^{T-1} [\ln \pi_\theta(a_t|s_t) + \ln P(s_{t+1}|s_t,a_t)]\\
&=\ln P(s_0) + \sum_{t=0}^{T-1} \ln \pi_\theta(a_t|s_t) + \sum_{t=0}^{T-1} \ln P(s_{t+1}|s_t,a_t)
\end{align}
$$
则：
$$
\begin{align}
\nabla_\theta \ln P(\tau|\theta)&=\nabla_\theta\left[\ln P(s_0) + \sum_{t=0}^{T-1} \ln \pi_\theta(a_t|s_t) + \sum_{t=0}^{T-1} \ln P(s_{t+1}|s_t,a_t)\right]\\
&=\nabla_\theta \sum_{t=0}^{T-1} \ln \pi_\theta(a_t|s_t) + \nabla_\theta \ln P(s_0) + \nabla_\theta \sum_{t=0}^{T-1} \ln P(s_{t+1}|s_t,a_t)\\
&=\nabla_\theta \sum_{t=0}^{T-1} \ln \pi_\theta(a_t|s_t)\\
&=\sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t|s_t)
\end{align}
$$
代入到**公式（1）**，有：
$$
\begin{align}
\nabla_\theta J(\theta)&=\mathbb{E}_{\tau \sim \pi_\theta}[R(\tau) \nabla_\theta \ln P(\tau|\theta)]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}\left[R(\tau) \sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t|s_t)\right] \tag{2}
\end{align}
$$
这已经是一个可用的公式了，只需要对轨迹进行采样计算就行了

对于其的直观理解是，如果一条轨迹的总折扣回报越大，则越发强化此轨迹上的所有动作选择

基于此理解，更好的方式是每个动作选择应该只为之后的轨迹负责

此公式还能进行进一步的优化：

由于不同的轨迹长短不一，我们定义序列$X_\tau(t)$:
$$
X_\tau(t)=
\begin{cases}
R(\tau) \nabla_\theta \ln \pi_\theta(a_t|s_t)=\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{T-1}\gamma^k r(k),&0 \le t \le T-1\\
0,&T \le t < \infty
\end{cases}
$$
将序列$X_\tau(t)$进行分解：
$$
X_\tau(t)=Past_\tau(t)+Future_\tau(t)
$$
其中：
$$
Past_\tau(t)=
\begin{cases}
\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k),&0 \le t \le T-1\\
0,&T \le t < \infty
\end{cases}
$$

$$
Future_\tau(t)=
\begin{cases}
\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=t}^{T-1} \gamma^k r(k),&0 \le t \le T-1\\
0,&T \le t < \infty
\end{cases}
$$

将$X_\tau(t)$代入**公式（2）**：
$$
\begin{align}
\nabla_\theta J(\theta)&=\mathbb{E}_{\tau \sim \pi_\theta}\left[R(\tau)\sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} R(\tau) \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}
\left[\sum_{t=0}^\infty X_\tau(t)\right]\\
&=\sum_{\tau}P(\tau|\theta)\sum_{t=0}^\infty X_\tau(t)\\
&=\sum_{\tau}\sum_{t=0}^\infty P(\tau|\theta)X_\tau(t)\\
&=\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)X_\tau(t)\\
&=\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)[Past_\tau(t)+Future_\tau(t)]\\
&=\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)Past_\tau(t)+\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)Future_\tau(t)\tag{3}
\end{align}
$$
而：
$$
\begin{align}
\sum_{\tau}P(\tau|\theta)Past_\tau(t)&=\sum_{\tau:T(\tau) > t}P(\tau|\theta)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k)\right]+\sum_{\tau:T(\tau) \le t}P(\tau|\theta)\cdot 0\\
&=\sum_{\tau:T(\tau) > t}P(\tau|\theta)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k)\right]\tag{4}
\end{align}
$$
接下来需要定义**吸收态**，对于任意的终止态，即使其外显属性不同，但仍将其视作同一个状态$s_{\perp}$，称作**吸收态**

**吸收态不同于任何一个中间态**，状态空间由$\{s_{\perp}\}$和所有中间态构成的集合并成

对于任意$T(\tau)>t$的轨迹，都可以将其看作三元组$(h_t,a_t,\tau')$，分别代表**轨迹的历史部分**、**当前动作**、**轨迹的剩余部分**，$h_t$不可包含吸收态

具体来说，轨迹集合和三元组集合存在一个双射，则求和可以分解到三个维度上，且其概率可以分解为三者的概率之积

对于任何一个$h_t$，其概率为多步的概率之积，任何长度不足的轨迹都没有$h_t$，这就隐式包含了对轨迹的长度约束

则：
$$
\begin{align}
\sum_{\tau:T(\tau) > t}P(\tau|\theta)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k)\right]&=\sum_{\tau:T(\tau) > t}P(h_t)\pi(a_t|h_t)P(\tau'|h_t,a_t)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k)\right]\\
&=\sum_{\tau:T(\tau) > t}P(h_t)\pi(a_t|s_t)P(\tau'|h_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k)\\
&=\sum_{h_t}\sum_{a_t}\sum_{\tau'}P(h_t)\pi(a_t|s_t)P(\tau'|h_t,a_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k)\\
&=\sum_{h_t}P(h_t)\left[\sum_{k=0}^{t-1} \gamma^k r(k)\right]\sum_{a_t}\pi(a_t|s_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{\tau'}P(\tau'|h_t,a_t)\\
&=\sum_{h_t}P(h_t)\left[\sum_{k=0}^{t-1} \gamma^k r(k)\right]\sum_{a_t}\pi(a_t|s_t)\nabla_\theta \ln \pi_\theta(a_t|s_t)\\
&=\sum_{h_t}P(h_t)\left[\sum_{k=0}^{t-1} \gamma^k r(k)\right]\sum_{a_t}\pi(a_t|s_t)\left[\frac{\nabla_\theta \pi_\theta(a_t|s_t)}{\pi_\theta(a_t|s_t)}\right]\\
&=\sum_{h_t}P(h_t)\left[\sum_{k=0}^{t-1} \gamma^k r(k)\right]\sum_{a_t}\nabla_\theta \pi_\theta(a_t|s_t)\\
&=\sum_{h_t}P(h_t)\left[\sum_{k=0}^{t-1} \gamma^k r(k)\right]\nabla_\theta \sum_{a_t}\pi_\theta(a_t|s_t)\\
&=\sum_{h_t}P(h_t)\left[\sum_{k=0}^{t-1} \gamma^k r(k)\right]\nabla_\theta 1\\
&=\sum_{h_t}P(h_t)\left[\sum_{k=0}^{t-1} \gamma^k r(k)\right] \cdot 0\\
&=0
\end{align}
$$
代回**公式（4）**，得：
$$
\begin{align}
\sum_{\tau}P(\tau|\theta)Past_\tau(t)&=\sum_{\tau:T(\tau) > t}P(\tau|\theta)\left[\nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=0}^{t-1} \gamma^k r(k)\right]\\
&=0
\end{align}
$$
再代回**公式（3）**，得：
$$
\begin{align}
\nabla_\theta J(\theta)&=\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)Past_\tau(t)+\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)Future_\tau(t)\\
&=\sum_{t=0}^\infty 0+\sum_{t=0}^\infty \sum_{\tau}P(\tau|\theta)Future_\tau(t)\\
&=\sum_{\tau}\sum_{t=0}^\infty P(\tau|\theta)Future_\tau(t)\\
&=\sum_{\tau} P(\tau|\theta) \sum_{t=0}^\infty Future_\tau(t)\\
&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^\infty Future_\tau(t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=t}^{T-1} \gamma^k r(k)+\sum_{t=T}^\infty 0\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \ln \pi_\theta(a_t|s_t)\sum_{k=t}^{T-1} \gamma^k r(k)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \left[\gamma^t \sum_{k=t}^{T-1} \gamma^{k-t} r(k)\right]\nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \gamma^t G(t) \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\tag{5}\\
\end{align}
$$
这就是**REINFORCE**算法用的公式，但是实际上对于**公式（5）**，它内部对于一个动作好坏的评价前面都有系数$\gamma^t$，但实际应用中，往往会去除这个系数

去除此系数后在数学上已经不是原本的梯度了，但是能避免轨迹中长远步骤的梯度消失

下面介绍**baseline**：

我们要证明：
$$
\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\right]=0 \tag{6}
$$
其中$b(s_t)$是一个只依赖于$s_t$的函数，我们定义序列$Y_\tau (t)$：
$$
Y_\tau (t)=
\begin{cases}
\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t),&0 \le t \le T-1\\
0,&T \le t < \infty
\end{cases}
$$
则：
$$
\begin{align}
\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\right]&=\sum_\tau P(\tau)\sum_{t=0}^{T-1}\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\\
&=\sum_\tau P(\tau)\sum_{t=0}^\infty Y_\tau (t)\\
&=\sum_{t=0}^\infty \sum_\tau P(\tau) Y_\tau (t)\tag{7}
\end{align}
$$
而：
$$
\begin{align}
\sum_\tau P(\tau)Y_\tau (t)&=\sum_{\tau:T(\tau)>t}P(\tau)\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)+\sum_{\tau:T(\tau) \le t}P(\tau) \cdot 0\\
&=\sum_{\tau:T(\tau)>t}P(\tau)\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\\
&=\sum_{\tau:T(\tau)>t}P(h_t)\pi(a_t|h_t)P(\tau'|h_t,a_t)\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\\
&=\sum_{\tau:T(\tau)>t}P(h_t)\pi(a_t|s_t)P(\tau'|h_t,a_t)\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\\
&=\sum_{h_t}\sum_{a_t}\sum_{\tau'}P(h_t)\pi(a_t|s_t)P(\tau'|h_t,a_t)\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\\
&=\sum_{h_t}P(h_t)\sum_{a_t}\pi(a_t|s_t)\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\sum_{\tau'}P(\tau'|h_t,a_t)\\
&=\sum_{h_t}P(h_t)\sum_{a_t}\pi(a_t|s_t)\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\\
&=\sum_{h_t}P(h_t)\sum_{a_t}\pi(a_t|s_t)\gamma^t b(s_t)\left[\frac{\nabla_\theta \pi_\theta (a_t|s_t)}{\pi_\theta (a_t|s_t)}\right]\\
&=\sum_{h_t}P(h_t)\sum_{a_t}\gamma^t b(s_t)\nabla_\theta \pi_\theta (a_t|s_t)\\
&=\gamma^t \sum_{h_t}P(h_t)\sum_{a_t}b(s_t)\nabla_\theta \pi_\theta (a_t|s_t)\\
\end{align}
$$
因为$b(s_t)$只依赖于$s_t$，而$h_t$确定后$s_t$就确定了，这意味着$h_t$确定了后$b(s_t)$就确定了，则：
$$
\begin{align}
\sum_\tau P(\tau)Y_\tau (t)&=\gamma^t \sum_{h_t}P(h_t)\sum_{a_t}b(s_t)\nabla_\theta \pi_\theta (a_t|s_t)\\
&=\gamma^t \sum_{h_t}P(h_t)b(s_t)\sum_{a_t}\nabla_\theta \pi_\theta (a_t|s_t)\\
&=\gamma^t \sum_{h_t}P(h_t)b(s_t)\nabla_\theta \sum_{a_t}\pi_\theta (a_t|s_t)\\
&=\gamma^t \sum_{h_t}P(h_t)b(s_t)\nabla_\theta 1\\
&=\gamma^t \sum_{h_t}P(h_t)b(s_t) \cdot 0\\
&=0
\end{align}
$$
将此结果代入**公式（7）**，得：
$$
\begin{align}
\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\right]&=\sum_{t=0}^\infty \sum_\tau P(\tau) Y_\tau (t)\\
&=\sum_{t=0}^\infty 0\\
&=0
\end{align}
$$
这就证明了**等式（6）**，由于其成立，我们可以在**公式（5）**中任意加减它：
$$
\begin{align}
\nabla_\theta J(\theta)&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \gamma^t G(t) \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \gamma^t G(t) \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]-\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t b(s_t)\nabla_\theta \ln \pi_\theta (a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \gamma^t [G(t)-b(s_t)] \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\tag{8}
\end{align}
$$
引入**baseline**是为了降低方差

对于同一状态不同动作，例如说状态$s$具有$a$、$b$俩个动作选择，假设对于二者的评判分别是$1005$和$1007$，这二者的差很小，但是因为$a$、$b$是两个不同的动作，一者的概率增大另一者需要相应减少，因此它们的梯度方向往往是相反的，在对$a$的梯度下降中走了很大的步子，在对$b$的梯度下降中则同样走了方向相反的很大的步子，这就导致了梯度方差大

**baseline**就是旨在解决这个问题的，对于上述例子，若**baseline**为$1000$，则二者都减去其之后变为$5$和$7$，再去做梯度下降，对$a$来说它迈了很小一步，对$b$来说它往反方向迈了很小一步，因此方差较小

引入了**baseline**之后的算法被称为**REINFORCE with baseline**

实际的算法中会拿一个网络用轨迹上的$G(t)$和其对应$s_t$对**baseline**进行拟合

**同样的，为了避免梯度消失，公式（8）中的$\gamma ^ t$往往会被省略**

### 关于AC

在**公式（8）**中，对于动作的好坏评判就由$\gamma ^t [G(t)-b(s_t)]$决定，如果用一个网络拟合**baseline**，这个网络参与了对于动作的好坏评判，似乎它就是所谓**Critic**，那么**REINFORCE with baseline**就是一种**AC**方法

但在现代**RL**的定义中，**REINFORCE with baseline**不被视作**AC**，原因在于其中对于动作好坏直接的评判项$G(t)$是采用整个轨迹的回报取得的，在**AC**中要求评判项必须使用**Critic**网络拟合，即使**REINFORCE with baseline**中的**baseline**本身要用**Critic**拟合，但是它不作为对动作的评判项存在，而仅仅是为减小方差而作为修正项存在，所以不被视作**AC**